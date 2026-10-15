/*
 * Index cache (miles-bench, VKD3D_BENCH_INDEX_CACHE=1).
 *
 * Miles Morales keeps all geometry in one 800 MiB CUSTOM write-combined L0 heap ("ManagedBuffer",
 * D3DBufferManager::InitBuffer 0x1d6e0e0), so ~70k indexed draws a second read ~515 MB/s of
 * indices across PCIe, while the distinct 64 KiB pages they touch add up to only ~30 MiB a second.
 * This keeps a VRAM copy of the index ranges draws use: the first draw that needs a run of source
 * pages gets the run copied into a VRAM buffer, and every draw over it binds the copy instead
 * (firstIndex rebased; shaders cannot observe firstIndex). Same index bytes, read locally.
 *
 * Copies are queued while the draw is recorded and go out on the memory transfer queue at the
 * start of the next ExecuteCommandLists. Each flush makes every queue wait for it
 * (vkd3d_add_wait_to_all_queues), and a list can only be submitted after it was recorded, so the
 * copy always lands before the draw reads it. The source is read at flush time, after the CPU has
 * written the geometry the draw was recorded for.
 *
 * The cache is a ring of 64 KiB pages with clock-style reuse: a new run takes the next pages whose
 * runs were not drawn in the last IDXC_REUSE_FRAMES presents (the game keeps at most a few frames in
 * flight), and runs still in use are skipped over, so the hot set stays resident.
 *
 * Invalidation: the engine writes geometry into the heap only through D3DBufferManager
 * (enginepatch.c hooks every writer and reports each written range after the write, plus every new
 * allocation). Each write stamps its source pages with a new generation; a cached run carries the
 * generation it was copied at and serves a draw only if none of the draw's pages were written
 * since. A GPU write into the heap (never seen: the ManagedBuffer is filled by CPU stores) turns the
 * cache off for good.
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

/* Page size and cache size: VKD3D_BENCH_INDEX_CACHE_PAGE_SHIFT (default 12 = 4 KiB; a draw's
 * indices are a few KiB, so 64 KiB pages made ~1600 draws a frame need ~100 MiB of cache and it
 * thrashed) and VKD3D_BENCH_INDEX_CACHE_MB (default 64). */
static unsigned int idxc_page_shift = 12;
static VkDeviceSize idxc_size = 64ull << 20;
#define IDXC_MIN_SOURCE (256ull << 20)
#define IDXC_REUSE_FRAMES 8

struct idxc_entry
{
    uint16_t pages;
    uint32_t slot; /* first cache page + 1, 0 = none */
};

struct idxc_copy
{
    VkDeviceSize src, dst, size;
};

static pthread_mutex_t idxc_lock = PTHREAD_MUTEX_INITIALIZER;
static int idxc_enabled = -1;
/* VKD3D_BENCH_INDEX_CACHE_VERIFY=1: the cache lives in CPU-readable memory and every 16th hit on a
 * run copied at least 4 presents ago is compared byte for byte with the source. A mismatch means a
 * geometry write the hooks did not report. Slow; for proving the cache lossless, not for play. */
static bool idxc_verify;
/* VKD3D_BENCH_INDEX_CACHE_FAILFAST=0 restores the full ring walk on every failed allocation (A/B). */
static bool idxc_failfast = true;

static struct
{
    struct d3d12_heap *heap;
    VkBuffer src_buffer;
    VkDeviceSize src_offset, src_size;
    struct vkd3d_memory_allocation cache;
    bool cache_ok;
    uint32_t cache_pages, cache_head;
    /* Per cache page: first page of the run covering it (-1 = free); at a run's first page also
     * the source page that owns it and the present count it was last drawn in. */
    int32_t *run_of;
    uint32_t *run_owner;
    uint64_t *run_last;
    struct idxc_entry *table;
    uint32_t *page_gen;  /* per source page: generation of the last write */
    uint32_t *run_gen;   /* per cache page, at a run's first page: generation it was copied at */
    uint32_t gen;
    const uint8_t *src_cpu;
    bool disabled;
    struct idxc_copy *pending;
    size_t pending_count, pending_size;
    uint64_t hits, misses, full, bytes, stale, invalidations, verified, mismatched;
    uint64_t *run_copied; /* per cache page, at a run's first page: present count of the copy */
    /* The present count of the last failed allocation and the smallest run size that failed in it:
     * within one present nothing becomes evictable (run_last only grows), so any run at least that
     * big fails again; it is refused without rescanning the ring. */
    uint64_t fail_frame;
    uint32_t fail_pages;
    uint64_t scan_steps;
    /* Allocation requests and failures by run size (pages: 1, 2, 3-4, 5-8, 9-16, 17-64, 65+), and the
     * largest free-or-evictable gap seen by a failed search (pages). Measurement only. */
    uint64_t req_size[7], fail_size[7];
    uint32_t fail_gap;
} idxc;

static unsigned int idxc_size_class(uint32_t pages)
{
    return pages <= 1 ? 0 : pages <= 2 ? 1 : pages <= 4 ? 2 : pages <= 8 ? 3 : pages <= 16 ? 4 : pages <= 64 ? 5 : 6;
}

static uint64_t idxc_frame;

void vkd3d_index_cache_present(void)
{
    __atomic_add_fetch(&idxc_frame, 1, __ATOMIC_RELAXED);
}

/* Presents so far (any swapchain), for bench probes that report per-frame rates. */
uint64_t vkd3d_bench_present_count(void)
{
    return __atomic_load_n(&idxc_frame, __ATOMIC_RELAXED);
}

bool vkd3d_index_cache_wanted(struct d3d12_device *device, const D3D12_HEAP_DESC *desc)
{
    char env[8];

    if (idxc_enabled < 0)
    {
        idxc_enabled = vkd3d_get_env_var("VKD3D_BENCH_INDEX_CACHE", env, sizeof(env)) && env[0] == '1';
        idxc_verify = vkd3d_get_env_var("VKD3D_BENCH_INDEX_CACHE_VERIFY", env, sizeof(env)) && env[0] == '1';
        idxc_failfast = !(vkd3d_get_env_var("VKD3D_BENCH_INDEX_CACHE_FAILFAST", env, sizeof(env)) && env[0] == '0');
        if (vkd3d_get_env_var("VKD3D_BENCH_INDEX_CACHE_PAGE_SHIFT", env, sizeof(env)) && atoi(env) >= 8 && atoi(env) <= 20)
            idxc_page_shift = atoi(env);
        if (vkd3d_get_env_var("VKD3D_BENCH_INDEX_CACHE_MB", env, sizeof(env)) && atoi(env) > 0)
            idxc_size = (VkDeviceSize)atoi(env) << 20;
    }
    if (!idxc_enabled || idxc.src_buffer ||
            desc->Properties.Type != D3D12_HEAP_TYPE_CUSTOM ||
            desc->Properties.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE ||
            desc->Properties.MemoryPoolPreference != D3D12_MEMORY_POOL_L0 ||
            (desc->Flags & D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS) != D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS ||
            desc->SizeInBytes < IDXC_MIN_SOURCE)
        return false;
    /* Copies stay exact only while every geometry write is reported (enginepatch.c); an unknown
     * exe build installs no hooks. With memory shared by CPU and GPU (UMA) the copy would be read
     * from the same memory as the original: nothing to gain. */
    if (!vkd3d_engine_geometry_hooked())
    {
        INFO("idxcache: the engine's geometry writes are not hooked; no index cache.\n");
        return false;
    }
    if (d3d12_device_is_uma(device, NULL))
    {
        INFO("idxcache: CPU and GPU share memory; no index cache.\n");
        return false;
    }
    return true;
}

/* Right after the geometry heap got its memory: allocate the VRAM cache now, at startup, so it
 * gets VRAM before the streamed textures fill the budget. */
void vkd3d_index_cache_register(struct d3d12_device *device, struct d3d12_heap *heap)
{
    struct vkd3d_allocate_heap_memory_info alloc_info;
    size_t src_pages;

    pthread_mutex_lock(&idxc_lock);
    if (!idxc.cache_ok)
    {
        memset(&alloc_info, 0, sizeof(alloc_info));
        alloc_info.heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (idxc_verify)
        {
            /* CPU-readable (cached system memory) but still usable as an index buffer. */
            alloc_info.heap_desc.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
            alloc_info.heap_desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
            alloc_info.heap_desc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        }
        alloc_info.heap_desc.SizeInBytes = idxc_size;
        alloc_info.heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        alloc_info.heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
        alloc_info.extra_allocation_flags = VKD3D_ALLOCATION_FLAG_INTERNAL_SCRATCH;
        alloc_info.vk_memory_priority = vkd3d_convert_to_vk_prio(D3D12_RESIDENCY_PRIORITY_HIGH);
        if (FAILED(vkd3d_allocate_heap_memory(device, &device->memory_allocator, &alloc_info, &idxc.cache)))
        {
            ERR("idxcache: failed to allocate the %"PRIu64" MiB cache.\n", (uint64_t)(idxc_size >> 20));
            pthread_mutex_unlock(&idxc_lock);
            return;
        }
        idxc.cache_pages = idxc_size >> idxc_page_shift;
        if (!(idxc.run_of = vkd3d_malloc(idxc.cache_pages * sizeof(*idxc.run_of))) ||
                !(idxc.run_owner = vkd3d_calloc(idxc.cache_pages, sizeof(*idxc.run_owner))) ||
                !(idxc.run_last = vkd3d_calloc(idxc.cache_pages, sizeof(*idxc.run_last))) ||
                !(idxc.run_gen = vkd3d_calloc(idxc.cache_pages, sizeof(*idxc.run_gen))) ||
                !(idxc.run_copied = vkd3d_calloc(idxc.cache_pages, sizeof(*idxc.run_copied))))
        {
            pthread_mutex_unlock(&idxc_lock);
            return;
        }
        memset(idxc.run_of, 0xff, idxc.cache_pages * sizeof(*idxc.run_of));
        idxc.cache_ok = true;
    }

    src_pages = (size_t)((heap->desc.SizeInBytes + (1u << idxc_page_shift) - 1) >> idxc_page_shift);
    if (!(idxc.table = vkd3d_calloc(src_pages, sizeof(*idxc.table))) ||
            !(idxc.page_gen = vkd3d_calloc(src_pages, sizeof(*idxc.page_gen))))
    {
        pthread_mutex_unlock(&idxc_lock);
        return;
    }
    idxc.heap = heap;
    idxc.src_buffer = heap->allocation.resource.vk_buffer;
    idxc.src_offset = heap->allocation.offset;
    idxc.src_size = heap->desc.SizeInBytes;
    idxc.src_cpu = heap->allocation.cpu_address;
    idxc.gen = 1;
    idxc.cache_head = 0;
    pthread_mutex_unlock(&idxc_lock);

    INFO("idxcache: source heap %p (%"PRIu64" MiB, memory type %u), cache %"PRIu64" MiB memory type %u, %u byte pages.\n",
            (void *)heap, (uint64_t)(idxc.src_size >> 20), heap->allocation.device_allocation.vk_memory_type,
            (uint64_t)(idxc_size >> 20), idxc.cache.device_allocation.vk_memory_type, 1u << idxc_page_shift);
}

void vkd3d_index_cache_unregister(struct d3d12_heap *heap)
{
    pthread_mutex_lock(&idxc_lock);
    if (idxc.heap == heap)
    {
        vkd3d_free(idxc.table);
        idxc.table = NULL;
        vkd3d_free(idxc.page_gen);
        idxc.page_gen = NULL;
        idxc.src_cpu = NULL;
        idxc.heap = NULL;
        idxc.src_buffer = VK_NULL_HANDLE;
        idxc.pending_count = 0;
    }
    pthread_mutex_unlock(&idxc_lock);
}

bool vkd3d_index_cache_is_source(VkBuffer buffer)
{
    return buffer && buffer == idxc.src_buffer;
}

/* The CPU finished writing [address, address + size) (enginepatch.c, after each geometry write). */
void vkd3d_index_cache_invalidate_cpu(const void *address, uint64_t size)
{
    const uint8_t *p = address;
    uint32_t first, last, i, g;

    if (!idxc.src_cpu || !size || p < idxc.src_cpu || p >= idxc.src_cpu + idxc.src_size)
        return;
    pthread_mutex_lock(&idxc_lock);
    if (idxc.page_gen && p >= idxc.src_cpu && p < idxc.src_cpu + idxc.src_size)
    {
        size = min(size, (uint64_t)(idxc.src_cpu + idxc.src_size - p));
        first = (uint32_t)((p - idxc.src_cpu) >> idxc_page_shift);
        last = (uint32_t)((p - idxc.src_cpu + size - 1) >> idxc_page_shift);
        g = ++idxc.gen;
        for (i = first; i <= last; i++)
            idxc.page_gen[i] = g;
        idxc.invalidations++;
    }
    pthread_mutex_unlock(&idxc_lock);
}

/* Cached copies could go stale behind the cache's back (a GPU write into the source heap, or a
 * geometry writer the engine hooks don't cover): no cache from here on. */
void vkd3d_index_cache_disable(const char *why)
{
    if (!idxc.disabled)
    {
        idxc.disabled = true;
        ERR("idxcache: %s, turning the index cache off.\n", why);
    }
}

/* Find `pages` contiguous cache pages for a new run, evicting runs not drawn recently. Returns the
 * first page, or -1 when everything in reach is still in use.
 *
 * A failed search walks the whole ring (thousands of runs). In heavy scenes the runs drawn in the
 * last IDXC_REUSE_FRAMES presents fill the ring and ~800 draws a frame fail; walking the ring for
 * each of them cost tens of ms per frame on the recording thread, with idxc_lock held. Once a
 * search has failed, larger-or-equal runs are refused for the rest of that present: the set of
 * evictable pages only shrinks within a present, so the walk would fail again. */
static int32_t idxc_alloc_locked(uint32_t pages)
{
    uint64_t frame = __atomic_load_n(&idxc_frame, __ATOMIC_RELAXED);
    uint32_t scanned = 0, head, i;
    int32_t run;

    idxc.req_size[idxc_size_class(pages)]++;
    if (pages > idxc.cache_pages)
        return -1;
    if (idxc_failfast && idxc.fail_frame == frame + 1 && pages >= idxc.fail_pages)
    {
        idxc.fail_size[idxc_size_class(pages)]++;
        return -1;
    }
    while (scanned < 2 * idxc.cache_pages)
    {
        idxc.scan_steps++;
        head = idxc.cache_head;
        if (head + pages > idxc.cache_pages)
        {
            scanned += idxc.cache_pages - head;
            idxc.cache_head = head = 0;
        }
        /* A recently drawn run in the way: step past it and keep looking. */
        for (i = head; i < head + pages; i++)
        {
            if ((run = idxc.run_of[i]) >= 0 && idxc.run_last[run] + IDXC_REUSE_FRAMES > frame)
                break;
        }
        if (i < head + pages)
        {
            while (i < idxc.cache_pages && idxc.run_of[i] == run)
                i++;
            scanned += i - head;
            idxc.cache_head = i;
            continue;
        }
        /* Evict every run overlapping [head, head + pages), including ones starting earlier: all
         * of its pages, from its first. (Comparing against run_of[i] while clearing freed only the
         * pages up to i: the rest stayed tied to the run's first page and, once a new run started
         * there, counted as part of it, so freed space looked drawn and the ring refused draws.) */
        for (i = head; i < head + pages; i++)
        {
            int32_t victim;

            if ((victim = idxc.run_of[i]) < 0)
                continue;
            if (idxc.table[idxc.run_owner[victim]].slot == (uint32_t)victim + 1)
                idxc.table[idxc.run_owner[victim]].slot = 0;
            for (run = victim; (uint32_t)run < idxc.cache_pages && idxc.run_of[run] == victim; run++)
                idxc.run_of[run] = -1;
        }
        for (i = head; i < head + pages; i++)
            idxc.run_of[i] = head;
        idxc.run_last[head] = frame;
        idxc.cache_head = head + pages;
        return head;
    }
    idxc.fail_size[idxc_size_class(pages)]++;
    {
        uint32_t gap = 0, best = 0;

        for (i = 0; i < idxc.cache_pages; i++)
        {
            run = idxc.run_of[i];
            if (run >= 0 && idxc.run_last[run] + IDXC_REUSE_FRAMES > frame)
                gap = 0;
            else if (++gap > best)
                best = gap;
        }
        idxc.fail_gap = max(idxc.fail_gap, best);
    }
    /* Stored as frame + 1 so that 0 (zero-initialized) never matches present 0. */
    if (idxc.fail_frame != frame + 1 || pages < idxc.fail_pages)
    {
        idxc.fail_frame = frame + 1;
        idxc.fail_pages = pages;
    }
    return -1;
}

/* The draw reads [offset, offset + size) of `buffer`. On success, bind *out_buffer at *out_offset
 * with firstIndex 0 instead. */
bool vkd3d_index_cache_lookup(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
        VkBuffer *out_buffer, VkDeviceSize *out_offset)
{
    VkDeviceSize rel, end, run_start;
    struct idxc_entry *e;
    uint32_t p0, pages;
    struct idxc_copy *c;

    if (!size)
        return false;

    pthread_mutex_lock(&idxc_lock);
    if (buffer != idxc.src_buffer || !idxc.table || offset < idxc.src_offset || idxc.disabled)
    {
        pthread_mutex_unlock(&idxc_lock);
        return false;
    }
    rel = offset - idxc.src_offset;
    end = rel + size;
    if (end > idxc.src_size)
    {
        pthread_mutex_unlock(&idxc_lock);
        return false;
    }
    p0 = rel >> idxc_page_shift;
    pages = ((end - 1) >> idxc_page_shift) - p0 + 1;
    e = &idxc.table[p0];

    if (e->slot && e->pages >= pages)
    {
        uint32_t i, g = idxc.run_gen[e->slot - 1];

        for (i = 0; i < pages; i++)
        {
            if (idxc.page_gen[p0 + i] > g)
                break;
        }
        if (i < pages)
        {
            e->slot = 0; /* written since it was copied: copy again */
            idxc.stale++;
        }
    }
    if (e->slot && e->pages >= pages)
    {
        idxc.run_last[e->slot - 1] = __atomic_load_n(&idxc_frame, __ATOMIC_RELAXED);
        idxc.hits++;
        if (idxc_verify && idxc.cache.cpu_address && !(idxc.hits & 15) &&
                idxc.run_copied[e->slot - 1] + 4 <= idxc.run_last[e->slot - 1])
        {
            const uint8_t *cached = (const uint8_t *)idxc.cache.cpu_address +
                    ((VkDeviceSize)(e->slot - 1) << idxc_page_shift) + (rel - ((VkDeviceSize)p0 << idxc_page_shift));

            idxc.verified++;
            if (memcmp(cached, idxc.src_cpu + rel, size))
            {
                if (idxc.mismatched++ < 10)
                    ERR("idxcache verify: %"PRIu64" bytes at heap offset %#"PRIx64" differ from the cached copy "
                            "(run gen %u, page gen %u).\n", (uint64_t)size, (uint64_t)rel,
                            idxc.run_gen[e->slot - 1], idxc.page_gen[p0]);
            }
        }
    }
    else
    {
        int32_t first;

        if (pages > UINT16_MAX || !vkd3d_array_reserve((void **)&idxc.pending, &idxc.pending_size,
                idxc.pending_count + 1, sizeof(*idxc.pending)) || (first = idxc_alloc_locked(pages)) < 0)
        {
            idxc.full++;
            pthread_mutex_unlock(&idxc_lock);
            return false;
        }
        e->pages = pages;
        e->slot = first + 1;
        idxc.run_owner[first] = p0;
        idxc.run_gen[first] = idxc.gen;
        idxc.run_copied[first] = __atomic_load_n(&idxc_frame, __ATOMIC_RELAXED);

        run_start = (VkDeviceSize)p0 << idxc_page_shift;
        c = &idxc.pending[idxc.pending_count++];
        c->src = idxc.src_offset + run_start;
        c->dst = idxc.cache.offset + ((VkDeviceSize)(e->slot - 1) << idxc_page_shift);
        c->size = min((VkDeviceSize)pages << idxc_page_shift, idxc.src_size - run_start);
        idxc.bytes += c->size;
        idxc.misses++;
    }

    *out_buffer = idxc.cache.resource.vk_buffer;
    *out_offset = idxc.cache.offset + ((VkDeviceSize)(e->slot - 1) << idxc_page_shift) +
            (rel - ((VkDeviceSize)p0 << idxc_page_shift));
    pthread_mutex_unlock(&idxc_lock);
    return true;
}

/* Start of every ExecuteCommandLists, before the memory transfer queue is flushed. */
void vkd3d_index_cache_flush(struct d3d12_device *device)
{
    size_t i;

    if (!idxc.pending_count)
        return;
    pthread_mutex_lock(&idxc_lock);
    for (i = 0; i < idxc.pending_count; i++)
        vkd3d_memory_transfer_queue_copy_buffer(&device->memory_transfers, idxc.src_buffer, idxc.pending[i].src,
                idxc.cache.resource.vk_buffer, idxc.pending[i].dst, idxc.pending[i].size, false);
    idxc.pending_count = 0;
    pthread_mutex_unlock(&idxc_lock);
}

bool vkd3d_index_cache_size_stats(uint64_t req[7], uint64_t fail[7], uint32_t *gap)
{
    unsigned int i;

    if (!idxc.cache_ok)
        return false;
    pthread_mutex_lock(&idxc_lock);
    for (i = 0; i < 7; i++)
    {
        req[i] = idxc.req_size[i];
        fail[i] = idxc.fail_size[i];
        idxc.req_size[i] = idxc.fail_size[i] = 0;
    }
    *gap = idxc.fail_gap;
    idxc.fail_gap = 0;
    pthread_mutex_unlock(&idxc_lock);
    return true;
}

bool vkd3d_index_cache_age_stats(uint32_t mib[7])
{
    uint64_t frame, pages[7] = {0}, age;
    uint32_t i, end, b;
    int32_t run;

    if (!idxc.cache_ok)
        return false;
    pthread_mutex_lock(&idxc_lock);
    frame = __atomic_load_n(&idxc_frame, __ATOMIC_RELAXED);
    for (i = 0; i < idxc.cache_pages; i = end)
    {
        end = i + 1;
        if ((run = idxc.run_of[i]) < 0)
        {
            pages[6]++;
            continue;
        }
        while (end < idxc.cache_pages && idxc.run_of[end] == run)
            end++;
        age = frame >= idxc.run_last[run] ? frame - idxc.run_last[run] : 0;
        b = age < 4 ? (uint32_t)age : (age < 8 ? 4 : 5);
        pages[b] += end - i;
    }
    pthread_mutex_unlock(&idxc_lock);
    for (b = 0; b < 7; b++)
        mib[b] = (uint32_t)((pages[b] << idxc_page_shift) >> 20);
    return true;
}

bool vkd3d_index_cache_take_stats(uint64_t *hits, uint64_t *misses, uint64_t *full, uint64_t *bytes,
        uint32_t *used_mib, uint64_t *stale, uint64_t *invalidations, uint64_t *verified, uint64_t *mismatched,
        uint64_t *scan_steps)
{
    if (!idxc.cache_ok)
        return false;
    pthread_mutex_lock(&idxc_lock);
    *hits = idxc.hits;
    *misses = idxc.misses;
    *full = idxc.full;
    *bytes = idxc.bytes;
    *used_mib = (uint32_t)(((uint64_t)idxc.cache_head << idxc_page_shift) >> 20); /* ring position */
    *stale = idxc.stale;
    *invalidations = idxc.invalidations;
    *verified = idxc.verified;
    *mismatched = idxc.mismatched;
    *scan_steps = idxc.scan_steps;
    idxc.scan_steps = 0;
    idxc.verified = 0; /* mismatched stays cumulative */
    idxc.hits = idxc.misses = idxc.full = idxc.bytes = idxc.stale = idxc.invalidations = 0;
    pthread_mutex_unlock(&idxc_lock);
    return true;
}
