/*
 * miles-bench: geometry heap mirror (VKD3D_BENCH_GEOMETRY_VRAM=1).
 *
 * Miles Morales keeps its streamed geometry in one 800 MiB CUSTOM heap (write-combined, L0 =
 * system memory, buffers only: the engine's "ManagedBuffer"). The CPU streams meshes into it and
 * never reads them back; the GPU never writes it. Every frame the GPU fetches indices (index
 * buffer views over the whole heap) and vertices (bindless raw/structured SRVs over the whole
 * heap, vertex pulling) straight out of system memory, several times per frame (depth prepass,
 * shadow cascades, main view), over PCIe.
 *
 * Here that heap lives in VRAM instead, and the game writes into a pageable shadow in system
 * memory that is watched with Windows write-watch (VirtualAlloc(MEM_WRITE_WATCH)). Before each
 * ExecuteCommandLists, the pages written since the last submission are copied shadow -> staging
 * ring -> VRAM on the memory transfer queue, which every queue's next submission waits for. The
 * GPU then reads the same bytes it read before, from VRAM. CPU writes reach the GPU no later than
 * the first submission after them, the same point a direct write to the old heap became usable.
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#define GEOMIRROR_MAX 4
#define GEOMIRROR_MIN_SIZE (256ull << 20)
#define GEOMIRROR_STAGING_SIZE (64ull << 20)
#define GEOMIRROR_SLICE_SHIFT 20  /* staging reuse is tracked per MiB */
#define GEOMIRROR_SLICES (GEOMIRROR_STAGING_SIZE >> GEOMIRROR_SLICE_SHIFT)
#define GEOMIRROR_MAX_COPY (8ull << 20)
#define GEOMIRROR_WATCH_BATCH 16384

static struct geomirror
{
    struct d3d12_heap *heap;
    uint8_t *shadow;
    VkDeviceSize size;
    VkBuffer vk_buffer;
    VkDeviceSize vk_offset;
} geomirrors[GEOMIRROR_MAX];
static unsigned int geomirror_count;
static pthread_mutex_t geomirror_lock = PTHREAD_MUTEX_INITIALIZER;

static struct vkd3d_memory_allocation geomirror_staging;
static bool geomirror_staging_ok;
static VkDeviceSize geomirror_staging_head;
static uint64_t geomirror_slice_value[GEOMIRROR_SLICES];
static void **geomirror_addrs;

static uint64_t geomirror_stat_bytes, geomirror_stat_runs, geomirror_stat_syncs, geomirror_stat_waits;
static uint64_t geomirror_stat_scan_ns;

static int geomirror_enabled = -1;
/* VKD3D_BENCH_GEOMIRROR_HOOKS=1: the engine's geometry write hooks (enginepatch.c) report every
 * written range, so the shadow needs no write watch (no fault per first write, no scan per
 * submission): reported ranges are copied at the next submission. */
static bool geomirror_hooks, geomirror_hooks_installed;
static struct { struct geomirror *m; VkDeviceSize offset, size; } *geomirror_dirty;
static size_t geomirror_dirty_count, geomirror_dirty_size;

static bool geomirror_is_geometry_heap(const D3D12_HEAP_DESC *desc)
{
    return desc->Properties.Type == D3D12_HEAP_TYPE_CUSTOM &&
            desc->Properties.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE &&
            desc->Properties.MemoryPoolPreference == D3D12_MEMORY_POOL_L0 &&
            (desc->Flags & D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS) == D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS &&
            !(desc->Flags & (D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH)) &&
            desc->SizeInBytes >= GEOMIRROR_MIN_SIZE;
}

bool vkd3d_geomirror_wanted(const D3D12_HEAP_DESC *desc)
{
    char env[8];

    if (geomirror_enabled < 0)
    {
        geomirror_enabled = vkd3d_get_env_var("VKD3D_BENCH_GEOMETRY_VRAM", env, sizeof(env)) && env[0] == '1';
        /* Without every writer hooked (enginepatch.c), fall back to the write watch. */
        geomirror_hooks = geomirror_hooks_installed &&
                vkd3d_get_env_var("VKD3D_BENCH_GEOMIRROR_HOOKS", env, sizeof(env)) && env[0] == '1';
    }
    return geomirror_enabled && geomirror_is_geometry_heap(desc);
}

/* VKD3D_BENCH_GEOMETRY_CACHED=1: back the geometry heap with CPU-cached (still coherent) system
 * memory instead of write-combined. The engine does read it on the CPU: a vertex decoder
 * (0x1c97030, 16-bit positions + 10:10:10 normals -> floats, ~55 call sites in 0x1c8a000-0x1c93600)
 * runs on the job threads and the render thread and loads straight from the heap. Write-combined
 * reads are uncached and never prefetched: ~844 cycles each, a quarter of the game's sampled
 * load-stall time, and four DRAM reads per 64-byte line. Same bytes, same GPU view; only the
 * CPU-side caching changes. */
bool vkd3d_geometry_cached_wanted(const D3D12_HEAP_DESC *desc)
{
    static int enabled = -1;
    char env[8];

    if (enabled < 0)
        enabled = vkd3d_get_env_var("VKD3D_BENCH_GEOMETRY_CACHED", env, sizeof(env)) && env[0] == '1';
    return enabled && geomirror_is_geometry_heap(desc);
}

static bool geomirror_init_staging_locked(struct d3d12_device *device)
{
    struct vkd3d_allocate_heap_memory_info alloc_info;

    if (geomirror_staging_ok)
        return true;

    if (!(geomirror_addrs = vkd3d_calloc(GEOMIRROR_WATCH_BATCH, sizeof(*geomirror_addrs))))
        return false;

    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.heap_desc.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
    alloc_info.heap_desc.SizeInBytes = GEOMIRROR_STAGING_SIZE;
    alloc_info.heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    alloc_info.heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
    alloc_info.extra_allocation_flags = VKD3D_ALLOCATION_FLAG_INTERNAL_SCRATCH;
    alloc_info.vk_memory_priority = vkd3d_convert_to_vk_prio(D3D12_RESIDENCY_PRIORITY_NORMAL);
    if (FAILED(vkd3d_allocate_heap_memory(device, &device->memory_allocator, &alloc_info, &geomirror_staging)) ||
            !geomirror_staging.cpu_address)
    {
        ERR("geomirror: failed to allocate the staging ring.\n");
        return false;
    }
    geomirror_staging_ok = true;
    return true;
}

/* Called right after the heap's memory was allocated (as DEFAULT, i.e. VRAM). Gives the heap a
 * write-watched shadow as its CPU address, so placed buffers map to the shadow. */
HRESULT vkd3d_geomirror_register(struct d3d12_device *device, struct d3d12_heap *heap)
{
    struct geomirror *m;
    void *shadow;

    pthread_mutex_lock(&geomirror_lock);
    if (geomirror_count == GEOMIRROR_MAX || !geomirror_init_staging_locked(device))
    {
        pthread_mutex_unlock(&geomirror_lock);
        return E_OUTOFMEMORY;
    }
    if (!(shadow = VirtualAlloc(NULL, (SIZE_T)heap->desc.SizeInBytes,
            MEM_COMMIT | MEM_RESERVE | (geomirror_hooks ? 0 : MEM_WRITE_WATCH), PAGE_READWRITE)))
    {
        ERR("geomirror: failed to allocate a %"PRIu64" byte shadow, error %#x.\n",
                (uint64_t)heap->desc.SizeInBytes, (unsigned int)GetLastError());
        pthread_mutex_unlock(&geomirror_lock);
        return E_OUTOFMEMORY;
    }

    m = &geomirrors[geomirror_count++];
    m->heap = heap;
    m->shadow = shadow;
    m->size = heap->desc.SizeInBytes;
    m->vk_buffer = heap->allocation.resource.vk_buffer;
    m->vk_offset = heap->allocation.offset;
    heap->allocation.cpu_address = shadow;
    pthread_mutex_unlock(&geomirror_lock);

    INFO("geomirror: heap %p, %"PRIu64" MiB, memory type %u, shadow %p, %s.\n", (void *)heap,
            (uint64_t)(m->size >> 20), heap->allocation.device_allocation.vk_memory_type, shadow,
            geomirror_hooks ? "synced from the engine's write hooks" : "write-watched");
    return S_OK;
}

void vkd3d_geomirror_unregister(struct d3d12_heap *heap)
{
    unsigned int i;

    pthread_mutex_lock(&geomirror_lock);
    for (i = 0; i < geomirror_count; i++)
    {
        if (geomirrors[i].heap != heap)
            continue;
        VirtualFree(geomirrors[i].shadow, 0, MEM_RELEASE);
        heap->allocation.cpu_address = NULL;
        geomirrors[i] = geomirrors[--geomirror_count];
        break;
    }
    pthread_mutex_unlock(&geomirror_lock);
}

/* Room for @size bytes in the staging ring, once the copies that last read it are done. */
static VkDeviceSize geomirror_staging_alloc_locked(struct d3d12_device *device, VkDeviceSize size)
{
    VkDeviceSize offset, s;

    if (geomirror_staging_head + size > GEOMIRROR_STAGING_SIZE)
        geomirror_staging_head = 0;
    offset = geomirror_staging_head;
    geomirror_staging_head += size;

    for (s = offset >> GEOMIRROR_SLICE_SHIFT; s <= (offset + size - 1) >> GEOMIRROR_SLICE_SHIFT; s++)
    {
        if (geomirror_slice_value[s] && !vkd3d_memory_transfer_queue_is_done(&device->memory_transfers,
                geomirror_slice_value[s]))
        {
            geomirror_stat_waits++;
            vkd3d_memory_transfer_queue_wait_value(&device->memory_transfers, geomirror_slice_value[s]);
        }
    }
    return offset;
}

static void geomirror_copy_run_locked(struct d3d12_device *device, struct geomirror *m,
        VkDeviceSize offset, VkDeviceSize size, bool *first)
{
    VkDeviceSize chunk, staging_offset, s;
    uint64_t value;

    /* Re-arm the watch on just this run before reading it: a write that lands after this marks
     * the page again and goes out with the next submission. (Resetting the whole heap instead
     * would make Wine mprotect all 800 MiB on every submission.) */
    if (!geomirror_hooks && ResetWriteWatch(m->shadow + offset, (SIZE_T)size))
        ERR("geomirror: ResetWriteWatch failed, error %#x.\n", (unsigned int)GetLastError());

    while (size)
    {
        chunk = min(size, GEOMIRROR_MAX_COPY);
        staging_offset = geomirror_staging_alloc_locked(device, chunk);
        memcpy((uint8_t *)geomirror_staging.cpu_address + staging_offset, m->shadow + offset, chunk);
        value = vkd3d_memory_transfer_queue_copy_buffer(&device->memory_transfers,
                geomirror_staging.resource.vk_buffer, geomirror_staging.offset + staging_offset,
                m->vk_buffer, m->vk_offset + offset, chunk, *first);
        *first = false;
        for (s = staging_offset >> GEOMIRROR_SLICE_SHIFT; s <= (staging_offset + chunk - 1) >> GEOMIRROR_SLICE_SHIFT; s++)
            geomirror_slice_value[s] = value;
        geomirror_stat_bytes += chunk;
        offset += chunk;
        size -= chunk;
    }
    geomirror_stat_runs++;
}

/* enginepatch.c hooked every geometry writer (at device creation, before any heap exists). */
void vkd3d_geomirror_hooks_installed(void)
{
    geomirror_hooks_installed = true;
}

bool vkd3d_geomirror_hooks_wanted(void)
{
    char env[8];

    return vkd3d_get_env_var("VKD3D_BENCH_GEOMETRY_VRAM", env, sizeof(env)) && env[0] == '1' &&
            vkd3d_get_env_var("VKD3D_BENCH_GEOMIRROR_HOOKS", env, sizeof(env)) && env[0] == '1';
}

/* The CPU finished writing [address, address + size) (enginepatch.c, after each geometry write). */
void vkd3d_geomirror_note_write(const void *address, uint64_t size)
{
    const uint8_t *p = address;
    struct geomirror *m;
    unsigned int k;

    if (!geomirror_hooks || !geomirror_count || !size)
        return;
    pthread_mutex_lock(&geomirror_lock);
    for (k = 0; k < geomirror_count; k++)
    {
        m = &geomirrors[k];
        if (p < m->shadow || p >= m->shadow + m->size)
            continue;
        size = min(size, (uint64_t)(m->shadow + m->size - p));
        if (geomirror_dirty_count && geomirror_dirty[geomirror_dirty_count - 1].m == m &&
                geomirror_dirty[geomirror_dirty_count - 1].offset +
                geomirror_dirty[geomirror_dirty_count - 1].size == (VkDeviceSize)(p - m->shadow))
        {
            geomirror_dirty[geomirror_dirty_count - 1].size += size; /* contiguous with the last one */
        }
        else if (vkd3d_array_reserve((void **)&geomirror_dirty, &geomirror_dirty_size,
                geomirror_dirty_count + 1, sizeof(*geomirror_dirty)))
        {
            geomirror_dirty[geomirror_dirty_count].m = m;
            geomirror_dirty[geomirror_dirty_count].offset = p - m->shadow;
            geomirror_dirty[geomirror_dirty_count++].size = size;
        }
        break;
    }
    pthread_mutex_unlock(&geomirror_lock);
}

/* Called at the start of every ExecuteCommandLists, before the memory transfer queue is flushed. */
void vkd3d_geomirror_sync(struct d3d12_device *device)
{
    VkDeviceSize run_start, run_end, page;
    ULONG_PTR count, i;
    struct geomirror *m;
    uint64_t t0, t1;
    unsigned int k;
    ULONG granularity;
    bool first = true;

    static unsigned int every, calls;
    char env[16];

    if (!geomirror_count)
        return;

    /* Experiment only (VKD3D_BENCH_GEOMIRROR_SYNC_EVERY=N): scan on every Nth submission. Not
     * correct in general (a write can reach the GPU N-1 submissions late); for measuring the GPU
     * side in scenes that write nothing to the heap after loading. */
    if (!every)
        every = vkd3d_get_env_var("VKD3D_BENCH_GEOMIRROR_SYNC_EVERY", env, sizeof(env)) && atoi(env) > 1 ? atoi(env) : 1;
    if (every > 1 && __atomic_fetch_add(&calls, 1, __ATOMIC_RELAXED) % every)
        return;

    if (geomirror_hooks)
    {
        size_t d;

        if (!__atomic_load_n(&geomirror_dirty_count, __ATOMIC_RELAXED))
            return;
        pthread_mutex_lock(&geomirror_lock);
        geomirror_stat_syncs++;
        for (d = 0; d < geomirror_dirty_count; d++)
            geomirror_copy_run_locked(device, geomirror_dirty[d].m, geomirror_dirty[d].offset,
                    geomirror_dirty[d].size, &first);
        geomirror_dirty_count = 0;
        pthread_mutex_unlock(&geomirror_lock);
        return;
    }

    pthread_mutex_lock(&geomirror_lock);
    geomirror_stat_syncs++;
    for (k = 0; k < geomirror_count; k++)
    {
        m = &geomirrors[k];
        for (;;)
        {
            /* Scan only; each dirty run is re-armed right before it is copied. */
            count = GEOMIRROR_WATCH_BATCH;
            t0 = vkd3d_get_current_time_ns();
            i = GetWriteWatch(0, m->shadow, (SIZE_T)m->size, geomirror_addrs, &count, &granularity);
            t1 = vkd3d_get_current_time_ns();
            geomirror_stat_scan_ns += t1 - t0;
            if (i)
            {
                ERR("geomirror: GetWriteWatch failed, error %#x.\n", (unsigned int)GetLastError());
                break;
            }
            if (!count)
                break;

            run_start = run_end = 0;
            for (i = 0; i < count; i++)
            {
                page = (uint8_t *)geomirror_addrs[i] - m->shadow;
                if (run_end && page == run_end)
                {
                    run_end += granularity;
                    continue;
                }
                if (run_end)
                    geomirror_copy_run_locked(device, m, run_start, run_end - run_start, &first);
                run_start = page;
                run_end = page + granularity;
            }
            if (run_end)
                geomirror_copy_run_locked(device, m, run_start, run_end - run_start, &first);

            if (count < GEOMIRROR_WATCH_BATCH)
                break;
        }
    }
    pthread_mutex_unlock(&geomirror_lock);
}

void vkd3d_geomirror_take_stats(uint64_t *bytes, uint64_t *runs, uint64_t *syncs, uint64_t *waits,
        uint64_t *scan_ns)
{
    pthread_mutex_lock(&geomirror_lock);
    *bytes = geomirror_stat_bytes;
    *runs = geomirror_stat_runs;
    *syncs = geomirror_stat_syncs;
    *waits = geomirror_stat_waits;
    *scan_ns = geomirror_stat_scan_ns;
    geomirror_stat_bytes = geomirror_stat_runs = geomirror_stat_syncs = geomirror_stat_waits = 0;
    geomirror_stat_scan_ns = 0;
    pthread_mutex_unlock(&geomirror_lock);
}

bool vkd3d_geomirror_active(void)
{
    return geomirror_count != 0;
}
