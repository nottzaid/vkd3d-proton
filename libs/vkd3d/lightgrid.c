/*
 * Lazy light grid (miles-bench, VKD3D_BENCH_LIGHTGRID_SPARSE=1).
 *
 * LightGridManager::Init (0x1c2d800) creates the "Light Grid Data Buffer" as a placed buffer at
 * offset 0 of a heap of its own: n = min(requested, 0x1fff) pages of 64 KiB, 512 MiB of VRAM. The
 * engine hands out pages 0..n-2 from a 3-level bitmap allocator (struct at 0x6a453a8, initialised
 * by 0x2164100 before the buffer is created; bit p of the last level, MSB first, is page p,
 * set = allocated; freed by 0x2163f20). Allocation is first fit: in the bench scene 1732 pages were
 * in use and the highest was page 1731. Page n-1 is the "null page": every "Light Grid Lookup"
 * entry starts out pointing at it, and the GPU only ever reaches other pages through lookup
 * entries the CPU writes after allocating them.
 *
 * Here that heap gets no memory of its own and the buffer is created as a sparse (reserved) buffer.
 * The null page and pages 0..bound-1 are backed by VRAM: bound starts one margin
 * (VKD3D_BENCH_LIGHTGRID_MARGIN pages, default 512) past the most pages earlier sessions of this
 * build had in use (learned.c), or at every page, as the original heap, before anything was learned
 * (VKD3D_BENCH_LIGHTGRID_INITIAL overrides it), and grows in 4 MiB steps to stay a margin ahead of
 * the highest allocated page. Every
 * ExecuteCommandLists first checks the bitmap and, if more pages are needed, backs them and waits
 * for the bind on the CPU before anything is submitted, on any queue. A list can only use a page
 * after the CPU allocated it, so every page a list can address is backed when the list reaches the
 * GPU.
 *
 * Growth needs free VRAM at a time the game's texture streamer has usually filled it, so the
 * budget the game reads (IDXGIAdapter3::QueryVideoMemoryInfo, local segment) is lowered by twice
 * the margin plus any shortfall: the streamer keeps that much VRAM free (it places textures it
 * cannot fit in system memory, same textures, sampled from there). In the open world the game goes
 * over that budget anyway (usage ~2480 MiB against ~2250, play-1013/1014) and growth then finds
 * no VRAM; the pages it could not back were read as zeros, lights missing (a visibly dark area in
 * the chase, user report 2026-10-14). Hence an initial backing that covers the most pages the game
 * has been seen to use, measured on this build rather than guessed. Growth would fall back to
 * system memory if the buffer allowed it, but on this GPU sparse buffers only take the VRAM type
 * (memory types 0x80). A failed growth is retried every 20 ms; a page in use that is not backed is
 * a loss and is counted and logged as one.
 *
 * Contents are those of the original: the original heap is a fresh, driver-zeroed allocation, the
 * backing chunks are fresh driver-zeroed allocations too (NVIDIA zeroes new device memory; vkd3d's
 * allocator relies on the same, vkd3d_driver_can_zero_clear_alloc), and backed pages stay backed
 * for the life of the buffer, so a page that is freed and handed out again keeps its old contents
 * exactly as before. Pages that are never backed were never allocated, so nothing writes them and
 * nothing reads them; with residencyNonResidentStrict a stray read would see zeros, which is what
 * the original's never-written pages hold.
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#define LIGHTGRID_PAGE 0x10000u
#define LIGHTGRID_CHUNK_PAGES 64u
#define LIGHTGRID_RETRY_NS 20000000ull
#define LIGHTGRID_BITMAP_RVA 0x6a453a8u
#define LIGHTGRID_EXE_TIMESTAMP 0x6374c7f4u /* MilesMorales.exe v1.1116.0.0 */

/* The engine's bitmap allocator (0x2164100). */
struct lightgrid_bitmap
{
    uint64_t *level0;
    uint64_t *level1;
    uint64_t *level2;
    int32_t count;
    int32_t used;
    int32_t highest;
};

static int lightgrid_enabled = -1;
/* Initial backing: in the open world the engine reached page 4341 (the Rhino chase) while the game
 * itself ran over its VRAM budget, so growth found no VRAM there, and on NVIDIA the sparse buffer
 * only takes VRAM-only memory types (no system memory fallback). So the backing made at creation,
 * while VRAM is still free, covers the most pages earlier sessions used (learned.c); with nothing
 * learned yet it is the whole buffer. lightgrid_initial: VKD3D_BENCH_LIGHTGRID_INITIAL, 0 = unset. */
static uint32_t lightgrid_margin = 512, lightgrid_initial;
static pthread_mutex_t lightgrid_lock = PTHREAD_MUTEX_INITIALIZER;

static struct
{
    const struct lightgrid_bitmap *bitmap;
    struct d3d12_device *device;
    struct d3d12_resource *resource;
    uint32_t pages;           /* n: buffer pages, the null page included */
    uint32_t bound;           /* pages 0..bound-1 are backed */
    uint32_t wanted;          /* pages the last growth aimed for */
    uint32_t memory_type;
    uint32_t type_bits;       /* the buffer's memory types (system memory fallback) */
    uint32_t fallback_type;   /* the system memory type last used for growth; memoryTypeCount: none yet */
    VkDeviceMemory *chunks;
    size_t chunk_count, chunk_size;
    VkDeviceMemory null_page;
    VkFence fence;
    bool broken;              /* setup failed: nothing is backed beyond what is */
    bool in_loss;             /* an allocated page is not backed (frametrace E lines mark changes) */
    uint64_t retry_at;
    uint32_t highest_seen;
    uint64_t grows, grow_ns, failed_allocs, unbacked_checks;
    uint32_t sysmem_pages;
} lightgrid;

static bool lightgrid_wanted(void)
{
    char env[16];

    if (lightgrid_enabled < 0)
    {
        lightgrid_enabled = vkd3d_get_env_var("VKD3D_BENCH_LIGHTGRID_SPARSE", env, sizeof(env)) && atoi(env) == 1;
        if (vkd3d_get_env_var("VKD3D_BENCH_LIGHTGRID_MARGIN", env, sizeof(env)) && atoi(env) > 0)
            lightgrid_margin = atoi(env);
        if (vkd3d_get_env_var("VKD3D_BENCH_LIGHTGRID_INITIAL", env, sizeof(env)) && atoi(env) > 0)
            lightgrid_initial = atoi(env);
    }
    return lightgrid_enabled;
}

static const struct lightgrid_bitmap *lightgrid_engine_bitmap(void)
{
    const IMAGE_NT_HEADERS64 *nt;
    const IMAGE_DOS_HEADER *dos;
    const uint8_t *image;

    if (!(image = (const uint8_t *)GetModuleHandleW(NULL)))
        return NULL;
    dos = (const IMAGE_DOS_HEADER *)image;
    nt = (const IMAGE_NT_HEADERS64 *)(image + dos->e_lfanew);
    if (nt->FileHeader.TimeDateStamp != LIGHTGRID_EXE_TIMESTAMP)
        return NULL;
    return (const struct lightgrid_bitmap *)(image + LIGHTGRID_BITMAP_RVA);
}

/* The light grid heap: DEFAULT, buffers only, exactly (bitmap pages + 1) x 64 KiB, created right
 * after the engine set up the bitmap for that many pages. */
/* Sparse buffers need sparse binding, buffer residency and vkd3d's internal sparse queue (which does
 * the binds); without them the light grid and skinning heaps keep their memory. */
bool vkd3d_bench_sparse_buffers(struct d3d12_device *device)
{
    const VkPhysicalDeviceFeatures *features = &device->device_info.features2.features;

    return device->internal_sparse_queue && features->sparseBinding && features->sparseResidencyBuffer;
}

bool vkd3d_lightgrid_heap_wanted(struct d3d12_device *device, const D3D12_HEAP_DESC *desc)
{
    const struct lightgrid_bitmap *bitmap;

    if (!lightgrid_wanted() || lightgrid.resource)
        return false;
    if (desc->Properties.Type != D3D12_HEAP_TYPE_DEFAULT ||
            (desc->Flags & (D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES)) !=
            (D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES))
        return false;
    if (!(bitmap = lightgrid_engine_bitmap()) || !bitmap->level2 || bitmap->count < 64 ||
            desc->SizeInBytes != (uint64_t)(bitmap->count + 1) * LIGHTGRID_PAGE)
        return false;
    if (!vkd3d_bench_sparse_buffers(device))
    {
        INFO("light grid: no sparse buffers; the heap keeps its memory.\n");
        return false;
    }
    INFO("light grid: heap of %u pages matches the engine's page bitmap; it gets no memory, "
            "its buffer is backed page by page.\n", bitmap->count + 1);
    return true;
}

static int32_t lightgrid_highest(void)
{
    const uint64_t *words = lightgrid.bitmap->level2;
    int32_t w;
    uint64_t v;

    for (w = (int32_t)((lightgrid.pages - 2) >> 6); w >= 0; w--)
    {
        if ((v = vkd3d_atomic_uint64_load_explicit((uint64_t *)&words[w], vkd3d_memory_order_relaxed)))
            return w * 64 + 63 - __builtin_ctzll(v); /* MSB first: the lowest set bit is the highest page */
    }
    return -1;
}

static bool lightgrid_bind(VkDeviceSize offset, VkDeviceSize size, VkDeviceMemory memory)
{
    const struct vkd3d_vk_device_procs *vk_procs = &lightgrid.device->vk_procs;
    struct d3d12_device *device = lightgrid.device;
    VkSparseBufferMemoryBindInfo buffer_info;
    VkBindSparseInfo bind_info;
    VkSparseMemoryBind bind;
    VkQueue vk_queue;
    VkResult vr;

    memset(&bind, 0, sizeof(bind));
    bind.resourceOffset = offset;
    bind.size = size;
    bind.memory = memory;
    buffer_info.buffer = lightgrid.resource->res.vk_buffer;
    buffer_info.bindCount = 1;
    buffer_info.pBinds = &bind;
    memset(&bind_info, 0, sizeof(bind_info));
    bind_info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    bind_info.bufferBindCount = 1;
    bind_info.pBufferBinds = &buffer_info;

    if (!(vk_queue = vkd3d_queue_acquire(device->internal_sparse_queue)))
        return false;
    vr = VK_CALL(vkQueueBindSparse(vk_queue, 1, &bind_info, lightgrid.fence));
    vkd3d_queue_release(device->internal_sparse_queue);
    if (vr >= 0)
        vr = VK_CALL(vkWaitForFences(device->vk_device, 1, &lightgrid.fence, VK_TRUE, UINT64_MAX));
    VK_CALL(vkResetFences(device->vk_device, 1, &lightgrid.fence));
    if (vr < 0)
        ERR("light grid: sparse bind of %#"PRIx64"+%#"PRIx64" failed, vr %d.\n", offset, size, vr);
    return vr >= 0;
}

static VkDeviceMemory lightgrid_alloc(VkDeviceSize size, uint32_t type)
{
    const struct vkd3d_vk_device_procs *vk_procs = &lightgrid.device->vk_procs;
    VkMemoryAllocateInfo info;
    VkDeviceMemory memory;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    info.memoryTypeIndex = type;
    if (VK_CALL(vkAllocateMemory(lightgrid.device->vk_device, &info, NULL, &memory)) < 0)
        return VK_NULL_HANDLE;
    return memory;
}

static uint32_t lightgrid_target(int32_t highest);

/* Back pages up to (not including) target. Called with lightgrid_lock held. `demand`: pages in use
 * are about to reach the backing (not the backing a new buffer starts with). */
static void lightgrid_grow(uint32_t target, bool demand)
{
    const struct vkd3d_vk_device_procs *vk_procs = &lightgrid.device->vk_procs;
    uint64_t t0 = vkd3d_get_current_time_ns();
    uint32_t count, bound0 = lightgrid.bound, sysmem0 = lightgrid.sysmem_pages;
    VkDeviceMemory memory;
    bool failed = false;

    lightgrid.wanted = max(lightgrid.wanted, target);
    while (lightgrid.bound < target)
    {
        count = min(LIGHTGRID_CHUNK_PAGES, lightgrid.pages - 1 - lightgrid.bound);
        if (!vkd3d_array_reserve((void **)&lightgrid.chunks, &lightgrid.chunk_size,
                lightgrid.chunk_count + 1, sizeof(*lightgrid.chunks)))
            memory = VK_NULL_HANDLE;
        else if (!(memory = lightgrid_alloc((VkDeviceSize)count * LIGHTGRID_PAGE, lightgrid.memory_type)) &&
                (lightgrid.fallback_type = vkd3d_bench_sysmem_alloc(lightgrid.device, lightgrid.type_bits,
                (VkDeviceSize)count * LIGHTGRID_PAGE, &memory)) < lightgrid.device->memory_properties.memoryTypeCount)
        {
            if (!lightgrid.sysmem_pages)
                ERR("light grid: VRAM is full; backing pages from %u on with system memory (type %u).\n",
                        lightgrid.bound, lightgrid.fallback_type);
            lightgrid.sysmem_pages += count;
        }
        if (!memory)
        {
            if (!(lightgrid.failed_allocs++ % 50))
                ERR("light grid: no VRAM for pages %u-%u yet (%"PRIu64" failed attempts); retrying.\n",
                        lightgrid.bound, target - 1, lightgrid.failed_allocs);
            lightgrid.retry_at = t0 + LIGHTGRID_RETRY_NS;
            failed = true;
            break;
        }
        if (!lightgrid_bind((VkDeviceSize)lightgrid.bound * LIGHTGRID_PAGE, (VkDeviceSize)count * LIGHTGRID_PAGE, memory))
        {
            VK_CALL(vkFreeMemory(lightgrid.device->vk_device, memory, NULL));
            lightgrid.retry_at = t0 + LIGHTGRID_RETRY_NS;
            failed = true;
            break;
        }
        lightgrid.chunks[lightgrid.chunk_count++] = memory;
        lightgrid.bound += count;
    }
    if (failed)
    {
        /* Only what the pages in use need stays wanted, so the budget hook does not hide a whole
         * initial backing that did not fit for the rest of the session. A growth the game needs
         * that found no VRAM is a failed VRAM allocation like any other: the streamer is told to
         * make room (vkd3d_bench_hold_raise). */
        lightgrid.wanted = max(lightgrid.bound, lightgrid_target(lightgrid_highest()));
        if (demand)
            vkd3d_bench_hold_raise((VkDeviceSize)min(LIGHTGRID_CHUNK_PAGES, target - lightgrid.bound) * LIGHTGRID_PAGE);
    }
    lightgrid.grows++;
    lightgrid.grow_ns += vkd3d_get_current_time_ns() - t0;
    vkd3d_frametrace_event("lightgrid grow %s %.3f ms pages %u -> %u, target %u, highest %u, system memory +%u pages",
            failed ? "fail" : "ok", (vkd3d_get_current_time_ns() - t0) / 1e6, bound0, lightgrid.bound, target,
            lightgrid.highest_seen, lightgrid.sysmem_pages - sysmem0);
    INFO("light grid: %u of %u pages backed (%u MiB, %u pages in system memory), highest allocated page %u, "
            "%.2f ms in %"PRIu64" grows.\n", lightgrid.bound + 1, lightgrid.pages, (lightgrid.bound + 1) >> 4,
            lightgrid.sysmem_pages, lightgrid.highest_seen, lightgrid.grow_ns / 1e6, lightgrid.grows);
}

static uint32_t lightgrid_target(int32_t highest)
{
    return min((uint32_t)(highest + 1) + lightgrid_margin, lightgrid.pages - 1);
}

HRESULT vkd3d_lightgrid_create_placed(struct d3d12_device *device, struct d3d12_heap *heap,
        uint64_t heap_offset, const D3D12_RESOURCE_DESC1 *desc, D3D12_RESOURCE_STATES initial_state,
        struct d3d12_resource **resource)
{
    const struct vkd3d_vk_device_procs *vk_procs = &device->vk_procs;
    VkMemoryRequirements requirements;
    struct d3d12_resource *object;
    VkFenceCreateInfo fence_info;
    uint32_t pages, type_bits, initial;
    const char *source = "setup failed";
    uint64_t learned = 0, reserve;
    HRESULT hr;

    pages = heap->desc.SizeInBytes / LIGHTGRID_PAGE;
    if (desc->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || heap_offset || desc->Width != heap->desc.SizeInBytes ||
            !device->internal_sparse_queue)
    {
        ERR("light grid: unexpected placement (dimension %u, offset %#"PRIx64", width %#"PRIx64") in the "
                "memoryless heap.\n", desc->Dimension, heap_offset, desc->Width);
        return E_INVALIDARG;
    }

    if (FAILED(hr = d3d12_resource_create_reserved(device, desc, initial_state, NULL, 0, NULL, &object)))
        return hr;
    if (!(object->flags & VKD3D_RESOURCE_RESERVED))
    {
        ERR("light grid: reserved buffer fell back to a committed one.\n");
        *resource = object;
        return S_OK;
    }
    d3d12_resource_wait_for_sparse_init(object);

    /* Plain VRAM, like the DEFAULT heap it replaces (not the host-visible BAR window). */
    VK_CALL(vkGetBufferMemoryRequirements(device->vk_device, object->res.vk_buffer, &requirements));
    type_bits = requirements.memoryTypeBits;
    for (lightgrid.memory_type = 0; lightgrid.memory_type < device->memory_properties.memoryTypeCount; lightgrid.memory_type++)
    {
        VkMemoryPropertyFlags flags = device->memory_properties.memoryTypes[lightgrid.memory_type].propertyFlags;

        if ((type_bits & (1u << lightgrid.memory_type)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            break;
    }
    lightgrid.type_bits = type_bits;
    lightgrid.fallback_type = device->memory_properties.memoryTypeCount;

    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    pthread_mutex_lock(&lightgrid_lock);
    lightgrid.bitmap = lightgrid_engine_bitmap();
    lightgrid.device = device;
    lightgrid.resource = object;
    lightgrid.pages = pages;
    lightgrid.bound = 0;
    if (lightgrid.memory_type == device->memory_properties.memoryTypeCount ||
            VK_CALL(vkCreateFence(device->vk_device, &fence_info, NULL, &lightgrid.fence)) < 0 ||
            !(lightgrid.null_page = lightgrid_alloc(LIGHTGRID_PAGE, lightgrid.memory_type)) ||
            !lightgrid_bind((VkDeviceSize)(pages - 1) * LIGHTGRID_PAGE, LIGHTGRID_PAGE, lightgrid.null_page))
    {
        ERR("light grid: setup failed (memory types %#x); the buffer stays unbacked.\n", type_bits);
        lightgrid.broken = true;
    }
    else
    {
        if (lightgrid_initial)
        {
            initial = min(lightgrid_initial, pages - 1);
            source = "VKD3D_BENCH_LIGHTGRID_INITIAL";
        }
        else if (vkd3d_bench_learned_get(VKD3D_BENCH_LEARNED_LIGHTGRID_PAGES, &learned) && learned &&
                vkd3d_bench_learned_get(VKD3D_BENCH_LEARNED_RESERVE_BYTES, &reserve))
        {
            initial = lightgrid_target((int32_t)min(learned, (uint64_t)pages - 1) - 1);
            source = "learned peak + margin";
        }
        else
        {
            /* As for the skinning buffer: lazy backing only where VRAM has run short before. */
            initial = pages - 1;
            source = learned ? "VRAM never ran short here: every page" : "nothing learned yet: every page";
        }
        lightgrid.highest_seen = max(lightgrid_highest(), 0);
        lightgrid_grow(max(initial, lightgrid_target(lightgrid_highest())), false);
    }
    pthread_mutex_unlock(&lightgrid_lock);

    INFO("light grid: \"Light Grid Data Buffer\" is sparse: %u pages, VA %#"PRIx64", memory type %u (types %#x), "
            "initial %u (%s), margin %u pages.\n", pages, object->res.va, lightgrid.memory_type,
            lightgrid.type_bits, lightgrid.bound, source, lightgrid_margin);
    *resource = object;
    return S_OK;
}

void vkd3d_lightgrid_update(void)
{
    uint32_t bound, target;
    int32_t highest;
    uint64_t now;

    if (!lightgrid.resource || lightgrid.broken)
        return;
    highest = lightgrid_highest();
    /* The peak is what later sessions back at creation; it is kept even when the backing is far ahead. */
    if (highest > (int32_t)vkd3d_atomic_uint32_load_explicit(&lightgrid.highest_seen, vkd3d_memory_order_relaxed))
    {
        pthread_mutex_lock(&lightgrid_lock);
        if (lightgrid.resource)
            lightgrid.highest_seen = max(lightgrid.highest_seen, (uint32_t)highest);
        pthread_mutex_unlock(&lightgrid_lock);
        vkd3d_bench_learned_raise(VKD3D_BENCH_LEARNED_LIGHTGRID_PAGES, (uint64_t)highest + 1);
    }
    bound = vkd3d_atomic_uint32_load_explicit(&lightgrid.bound, vkd3d_memory_order_acquire);
    if ((uint32_t)(highest + 1) + lightgrid_margin / 2 <= bound)
        return;

    pthread_mutex_lock(&lightgrid_lock);
    if (lightgrid.resource)
    {
        highest = lightgrid_highest();
        lightgrid.highest_seen = max(lightgrid.highest_seen, (uint32_t)max(highest, 0));
        target = lightgrid_target(highest);
        now = vkd3d_get_current_time_ns();
        if (target > lightgrid.bound && (now >= lightgrid.retry_at || highest >= (int32_t)lightgrid.bound))
            lightgrid_grow(target, true);
        if ((highest >= (int32_t)lightgrid.bound) != lightgrid.in_loss)
        {
            lightgrid.in_loss = !lightgrid.in_loss;
            vkd3d_frametrace_event("lightgrid loss %s page %d bound %u", lightgrid.in_loss ? "start" : "end",
                    highest, lightgrid.bound);
        }
        if (highest >= (int32_t)lightgrid.bound && !(lightgrid.unbacked_checks++ % 100))
            ERR("light grid: LOSS: page %d is in use but only %u pages are backed (%"PRIu64" submissions so far).\n",
                    highest, lightgrid.bound, lightgrid.unbacked_checks);
    }
    pthread_mutex_unlock(&lightgrid_lock);
}

/* VRAM the game should leave free for growth: what backing three margins past the highest page
 * ever allocated (at most the whole buffer) or the pages still wanted would add. The hold ramps up
 * from nothing as the pages in use come within three margins of the backed range and reaches two
 * margins when growth is due (it starts one margin ahead), so the streamer makes room in time; while
 * the backing is well ahead the game's budget is left as it is. */
uint64_t vkd3d_lightgrid_budget_reserve(void)
{
    uint32_t wanted, bound, highest, target;

    if (!lightgrid.resource || lightgrid.broken)
        return 0;
    wanted = vkd3d_atomic_uint32_load_explicit(&lightgrid.wanted, vkd3d_memory_order_relaxed);
    bound = vkd3d_atomic_uint32_load_explicit(&lightgrid.bound, vkd3d_memory_order_relaxed);
    highest = vkd3d_atomic_uint32_load_explicit(&lightgrid.highest_seen, vkd3d_memory_order_relaxed);
    target = max(min(highest + 3 * lightgrid_margin, lightgrid.pages - 1), wanted);
    return target > bound ? (uint64_t)(target - bound) * LIGHTGRID_PAGE : 0;
}

/* Called from d3d12_resource_destroy once the buffer's Vulkan objects are gone, which is when
 * vkd3d frees a placed resource's memory too. */
void vkd3d_lightgrid_resource_destroyed(struct d3d12_resource *resource)
{
    const struct vkd3d_vk_device_procs *vk_procs;
    size_t i;

    if (!resource || resource != lightgrid.resource)
        return;
    pthread_mutex_lock(&lightgrid_lock);
    vk_procs = &lightgrid.device->vk_procs;
    INFO("light grid: buffer destroyed; %u of %u pages were backed (%u in system memory), highest allocated page %u, "
            "%"PRIu64" failed allocations, %"PRIu64" submissions with an unbacked page in use.\n",
            lightgrid.bound + 1, lightgrid.pages, lightgrid.sysmem_pages, lightgrid.highest_seen,
            lightgrid.failed_allocs, lightgrid.unbacked_checks);
    for (i = 0; i < lightgrid.chunk_count; i++)
        VK_CALL(vkFreeMemory(lightgrid.device->vk_device, lightgrid.chunks[i], NULL));
    if (lightgrid.null_page)
        VK_CALL(vkFreeMemory(lightgrid.device->vk_device, lightgrid.null_page, NULL));
    if (lightgrid.fence)
        VK_CALL(vkDestroyFence(lightgrid.device->vk_device, lightgrid.fence, NULL));
    vkd3d_free(lightgrid.chunks);
    memset(&lightgrid, 0, sizeof(lightgrid));
    pthread_mutex_unlock(&lightgrid_lock);
}

/* The budget the game sees. DXVK's adapter implements IDXGIAdapter4 in one chain, so
 * QueryVideoMemoryInfo is vtable slot 14 (IUnknown 3, IDXGIObject 4, IDXGIAdapter 3, 1 and 2 one
 * each, then IDXGIAdapter3's RegisterHardwareContentProtectionTeardownStatusEvent and
 * UnregisterHardwareContentProtectionTeardownStatus). The vtable is shared by every adapter object
 * of that class, the game's included. */
struct lightgrid_video_memory_info
{
    uint64_t budget, current_usage, available_for_reservation, current_reservation;
};

typedef HRESULT (STDMETHODCALLTYPE *lightgrid_query_video_memory_info)(IUnknown *iface, UINT node,
        UINT segment_group, struct lightgrid_video_memory_info *info);
static lightgrid_query_video_memory_info lightgrid_query_original;
static uint64_t lightgrid_budget_queries;

/* The game's texture streamer fills the budget it is told and then some (it ran 70-95 MiB over it
 * in the fight), so on a card with no VRAM to spare it also takes the VRAM that vkd3d's own later
 * allocations (render targets, buffers, chunks) need; those then fail and land in system memory
 * (play-1015: 119-269 failed VRAM allocations, -6.4% fps, with only the growth margins held back).
 * So the budget the game is told also leaves out a learned hold: what earlier sessions of this build
 * found it takes (learned.c), raised at once when one of vkd3d's allocations of VRAM fails
 * (vkd3d_bench_hold_raise, from memtrace.c), so the streamer makes room in the same session. A raise
 * is the failed allocation's size (holding X back frees less than X: the streamer gives up only part
 * of it), and failures within the streamer's reaction time (2 s) after a raise add only what a larger
 * one needs beyond it, so a burst of failures counts once. It never shrinks within a session (the
 * streamer would reload what it just dropped). */
#define BENCH_HOLD_REACTION_NS 2000000000ull
static uint64_t bench_hold, bench_hold_window_start, bench_hold_window_max;
static pthread_mutex_t bench_hold_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t bench_hold_once = PTHREAD_ONCE_INIT;

static void bench_hold_init(void)
{
    uint64_t learned;
    char env[16];

    if (vkd3d_bench_learned_get(VKD3D_BENCH_LEARNED_HOLD_BYTES, &learned))
        bench_hold = learned;
    /* VKD3D_BENCH_BUDGET_HOLD_MB=N: at least N MiB (experiments). */
    if (vkd3d_get_env_var("VKD3D_BENCH_BUDGET_HOLD_MB", env, sizeof(env)) && *env)
        bench_hold = max(bench_hold, (uint64_t)atoi(env) << 20);
    INFO("light grid: %"PRIu64" MiB of VRAM held back from the game's budget besides growth room.\n", bench_hold >> 20);
}

/* A VRAM allocation of `size` bytes failed. */
void vkd3d_bench_hold_raise(uint64_t size)
{
    uint64_t now = vkd3d_get_current_time_ns(), raise, hold;

    pthread_once(&bench_hold_once, bench_hold_init);
    pthread_mutex_lock(&bench_hold_lock);
    if (now - bench_hold_window_start >= BENCH_HOLD_REACTION_NS)
    {
        bench_hold_window_start = now;
        bench_hold_window_max = 0;
    }
    raise = size > bench_hold_window_max ? size - bench_hold_window_max : 0;
    bench_hold_window_max = max(bench_hold_window_max, size);
    hold = __atomic_add_fetch(&bench_hold, raise, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&bench_hold_lock);
    if (!raise)
        return;
    vkd3d_bench_learned_raise(VKD3D_BENCH_LEARNED_HOLD_BYTES, hold);
    INFO("light grid: a %.1f MiB VRAM allocation failed; holding %.1f MiB back from the game now.\n",
            size / 1048576.0, hold / 1048576.0);
}

static HRESULT STDMETHODCALLTYPE lightgrid_query_video_memory_info_hook(IUnknown *iface, UINT node,
        UINT segment_group, struct lightgrid_video_memory_info *info)
{
    uint64_t reserve;
    HRESULT hr;

    pthread_once(&bench_hold_once, bench_hold_init);
    hr = lightgrid_query_original(iface, node, segment_group, info);
    if (SUCCEEDED(hr) && info && segment_group == 0 /* DXGI_MEMORY_SEGMENT_GROUP_LOCAL */ &&
            !vkd3d_bench_gpu_integrated() &&
            (reserve = max(vkd3d_lightgrid_budget_reserve() + vkd3d_skinbuf_budget_reserve(),
            __atomic_load_n(&bench_hold, __ATOMIC_RELAXED))))
    {
        info->budget = info->budget > reserve ? info->budget - reserve : 0;
        if (!(lightgrid_budget_queries++ % 3600))
            INFO("light grid: VRAM budget reported %"PRIu64" MiB (%"PRIu64" MiB held back), usage %"PRIu64" MiB.\n",
                    info->budget >> 20, reserve >> 20, info->current_usage >> 20);
    }
    return hr;
}

void vkd3d_lightgrid_hook_budget(IUnknown *adapter)
{
    static const GUID iid_adapter3 = {0x645967a4, 0x1392, 0x4310, {0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd}};
    IUnknown *adapter3;
    DWORD old_protect;
    void **vtbl;

    if ((!lightgrid_wanted() && !vkd3d_skinbuf_wanted()) || lightgrid_query_original || !adapter)
        return;
    if (FAILED(IUnknown_QueryInterface(adapter, &iid_adapter3, (void **)&adapter3)))
    {
        ERR("light grid: adapter has no IDXGIAdapter3; the VRAM budget is not adjusted.\n");
        return;
    }
    vtbl = *(void ***)adapter3;
    if (VirtualProtect(&vtbl[14], sizeof(void *), PAGE_READWRITE, &old_protect))
    {
        lightgrid_query_original = (lightgrid_query_video_memory_info)vtbl[14];
        vtbl[14] = (void *)lightgrid_query_video_memory_info_hook;
        VirtualProtect(&vtbl[14], sizeof(void *), old_protect, &old_protect);
        INFO("light grid: hooked QueryVideoMemoryInfo (vtable %p).\n", vtbl);
    }
    IUnknown_Release(adapter3);
}
