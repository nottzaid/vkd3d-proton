/*
 * Lazy skinning buffer (miles-bench, VKD3D_BENCH_SKIN_SPARSE=1).
 *
 * SkinningBuffer::InitD3DBuffers (0x1da3530, called from the skinning manager's init 0x1da39d0 for
 * the manager at 0x83ce300) creates one placed buffer at offset 0 of a heap of its own: CUSTOM,
 * CPU NOT_AVAILABLE, pool L1 (VRAM), buffers only, 196 MiB. The manager splits it into two frame
 * halves of H = ((total - frozen) / 2) & ~63 bytes (plus a "frozen" region after them, empty in
 * this build's settings) and hands out space from each half with the engine's lock-free bump
 * allocator: {u64 base, u32 size, u32 used, u32 align} at manager + 0x10 + half * 0x80, allocated
 * by 0x2162ef0 (four call sites, all with the current frame's half) and reset at the end of a
 * render-thread frame (0x1da3200 -> 0x2162ff0). In play a half reaches ~11 MiB of its 98 MiB.
 *
 * Here that heap gets no memory of its own and the buffer is created as a sparse (reserved) buffer;
 * each half is backed from its start up to a bound that starts at VKD3D_BENCH_SKIN_INITIAL_MB
 * (default 32) and grows in 4 MiB chunks to stay VKD3D_BENCH_SKIN_MARGIN_MB (default 8) ahead of
 * the highest byte ever allocated in it. The engine's calls to 0x2162ef0 and 0x2162f20 (the
 * aligned variant, which the skinning code does not use but which is patched the same way) are
 * redirected to skinbuf_alloc / skinbuf_alloc_aligned below: the same atomic operations on the
 * same allocator, and before an allocation from a skinning half is returned to its caller, the
 * range it covers is backed (synchronous bind). The GPU can only reach a range through a pointer
 * the allocator returned, so nothing it uses is ever unbacked. As a cross-check, every
 * ExecuteCommandLists compares both halves' `used` with their bounds; an allocation that bypassed
 * the hooks would show up there and is counted as a loss.
 *
 * Contents: the original heap is a fresh driver-zeroed allocation, the backing chunks are fresh
 * driver-zeroed allocations too, and backed ranges stay backed for the life of the buffer, so every
 * byte the engine can reach holds what it would hold in the original. VRAM for growth is kept free
 * through the DXGI budget hook in lightgrid.c (two margins, plus any shortfall), and the initial
 * backing covers what the open world needs while VRAM is full (as in lightgrid.c; a system memory
 * fallback is tried but this GPU's sparse buffers only take the VRAM type).
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#define SKIN_CHUNK (4u << 20)
#define SKIN_PAGE 0x10000u
#define SKIN_MGR_RVA 0x83ce300u
#define SKIN_TOTAL_RVA 0x84268e8u
#define SKIN_FROZEN_RVA 0x84268f0u
#define SKIN_EXE_TIMESTAMP 0x6374c7f4u /* MilesMorales.exe v1.1116.0.0 */
#define SKIN_RETRY_NS 20000000ull

/* The engine's bump allocator (0x2162fb0 init, 0x2162ef0 alloc, 0x2162f20 aligned alloc). */
struct engine_linear
{
    uint8_t *base;
    uint32_t size;
    uint32_t used;
    uint32_t align;
};

static int skin_enabled = -1;
/* 32 MiB per half at creation: the chase reaches 25.6 MiB while VRAM is full (play-1013), and the
 * sparse buffer only takes VRAM-only memory types, so the room has to be there from the start. */
static uint32_t skin_initial = 32u << 20, skin_margin = 8u << 20;
static pthread_mutex_t skin_lock = PTHREAD_MUTEX_INITIALIZER;

static struct
{
    struct engine_linear *half[2];
    struct d3d12_device *device;
    struct d3d12_resource *resource;
    uint32_t half_size;
    uint32_t bound[2];      /* bytes of each half backed from its start */
    uint32_t wanted[2];
    uint32_t highest[2];    /* end of the highest allocation seen in each half */
    uint32_t memory_type;
    uint32_t type_bits;     /* the buffer's memory types (system memory fallback) */
    uint32_t fallback_type; /* the system memory type last used for growth */
    uint32_t sysmem_bytes;
    VkDeviceMemory *chunks;
    size_t chunk_count, chunk_size;
    VkFence fence;
    bool broken;
    bool in_loss[2];        /* an allocation past the bound (frametrace E lines mark changes) */
    uint64_t retry_at;
    uint64_t allocs, grows, grow_ns, failed_allocs, losses, unbacked_checks;
} skin;

bool vkd3d_skinbuf_wanted(void)
{
    char env[16];

    if (skin_enabled < 0)
    {
        skin_enabled = vkd3d_get_env_var("VKD3D_BENCH_SKIN_SPARSE", env, sizeof(env)) && atoi(env) == 1;
        if (vkd3d_get_env_var("VKD3D_BENCH_SKIN_INITIAL_MB", env, sizeof(env)) && atoi(env) > 0)
            skin_initial = (uint32_t)atoi(env) << 20;
        if (vkd3d_get_env_var("VKD3D_BENCH_SKIN_MARGIN_MB", env, sizeof(env)) && atoi(env) > 0)
            skin_margin = (uint32_t)atoi(env) << 20;
    }
    return skin_enabled;
}

static uint8_t *skin_engine_image(void)
{
    const IMAGE_NT_HEADERS64 *nt;
    const IMAGE_DOS_HEADER *dos;
    uint8_t *image;

    if (!(image = (uint8_t *)GetModuleHandleW(NULL)))
        return NULL;
    dos = (const IMAGE_DOS_HEADER *)image;
    nt = (const IMAGE_NT_HEADERS64 *)(image + dos->e_lfanew);
    return nt->FileHeader.TimeDateStamp == SKIN_EXE_TIMESTAMP ? image : NULL;
}

/* The skinning heap: CUSTOM / NOT_AVAILABLE / L1, buffers only, exactly the size the manager's init
 * computes from the engine's settings, with no frozen region. */
bool vkd3d_skinbuf_heap_wanted(const D3D12_HEAP_DESC *desc)
{
    uint32_t total, frozen, half;
    uint8_t *image;

    if (!vkd3d_skinbuf_wanted() || skin.resource || !(image = skin_engine_image()))
        return false;
    if (desc->Properties.Type != D3D12_HEAP_TYPE_CUSTOM ||
            desc->Properties.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE ||
            desc->Properties.MemoryPoolPreference != D3D12_MEMORY_POOL_L1 ||
            (desc->Flags & (D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES)) !=
            (D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES))
        return false;
    total = *(const uint32_t *)(image + SKIN_TOTAL_RVA);
    frozen = *(const uint32_t *)(image + SKIN_FROZEN_RVA) & ~63u;
    half = ((total - frozen) >> 1) & 0x7fffffc0u;
    if (frozen || !half || desc->SizeInBytes != (uint64_t)half * 2 || (half % SKIN_PAGE))
        return false;
    INFO("skinning buffer: heap of %u MiB (two %u MiB frame halves) gets no memory; its buffer is backed "
            "as the engine allocates.\n", (unsigned int)(desc->SizeInBytes >> 20), half >> 20);
    return true;
}

static bool skin_bind(VkDeviceSize offset, VkDeviceSize size, VkDeviceMemory memory)
{
    const struct vkd3d_vk_device_procs *vk_procs = &skin.device->vk_procs;
    struct d3d12_device *device = skin.device;
    VkSparseBufferMemoryBindInfo buffer_info;
    VkBindSparseInfo bind_info;
    VkSparseMemoryBind bind;
    VkQueue vk_queue;
    VkResult vr;

    memset(&bind, 0, sizeof(bind));
    bind.resourceOffset = offset;
    bind.size = size;
    bind.memory = memory;
    buffer_info.buffer = skin.resource->res.vk_buffer;
    buffer_info.bindCount = 1;
    buffer_info.pBinds = &bind;
    memset(&bind_info, 0, sizeof(bind_info));
    bind_info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    bind_info.bufferBindCount = 1;
    bind_info.pBufferBinds = &buffer_info;

    if (!(vk_queue = vkd3d_queue_acquire(device->internal_sparse_queue)))
        return false;
    vr = VK_CALL(vkQueueBindSparse(vk_queue, 1, &bind_info, skin.fence));
    vkd3d_queue_release(device->internal_sparse_queue);
    if (vr >= 0)
        vr = VK_CALL(vkWaitForFences(device->vk_device, 1, &skin.fence, VK_TRUE, UINT64_MAX));
    VK_CALL(vkResetFences(device->vk_device, 1, &skin.fence));
    if (vr < 0)
        ERR("skinning buffer: sparse bind of %#"PRIx64"+%#"PRIx64" failed, vr %d.\n", offset, size, vr);
    return vr >= 0;
}

static VkDeviceMemory skin_alloc_memory(VkDeviceSize size, uint32_t type)
{
    const struct vkd3d_vk_device_procs *vk_procs = &skin.device->vk_procs;
    VkMemoryAllocateInfo info;
    VkDeviceMemory memory;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    info.memoryTypeIndex = type;
    if (VK_CALL(vkAllocateMemory(skin.device->vk_device, &info, NULL, &memory)) < 0)
        return VK_NULL_HANDLE;
    return memory;
}

/* Back half h up to (not including) byte `target`. Called with skin_lock held. */
static void skin_grow(unsigned int h, uint32_t target)
{
    const struct vkd3d_vk_device_procs *vk_procs = &skin.device->vk_procs;
    uint64_t t0 = vkd3d_get_current_time_ns();
    uint32_t count, bound0 = skin.bound[h];
    VkDeviceMemory memory;
    bool failed = false;

    target = min(target, skin.half_size);
    skin.wanted[h] = max(skin.wanted[h], target);
    while (skin.bound[h] < target)
    {
        count = min(SKIN_CHUNK, skin.half_size - skin.bound[h]);
        if (!vkd3d_array_reserve((void **)&skin.chunks, &skin.chunk_size, skin.chunk_count + 1, sizeof(*skin.chunks)))
            memory = VK_NULL_HANDLE;
        else if (!(memory = skin_alloc_memory(count, skin.memory_type)) &&
                (skin.fallback_type = vkd3d_bench_sysmem_alloc(skin.device, skin.type_bits, count, &memory)) <
                skin.device->memory_properties.memoryTypeCount)
        {
            if (!skin.sysmem_bytes)
                ERR("skinning buffer: VRAM is full; backing half %u from %#x on with system memory (type %u).\n",
                        h, skin.bound[h], skin.fallback_type);
            skin.sysmem_bytes += count;
        }
        if (!memory)
        {
            if (!(skin.failed_allocs++ % 50))
                ERR("skinning buffer: no VRAM to back half %u bytes %#x-%#x yet (%"PRIu64" failed attempts).\n",
                        h, skin.bound[h], target, skin.failed_allocs);
            skin.retry_at = t0 + SKIN_RETRY_NS;
            failed = true;
            break;
        }
        if (!skin_bind((VkDeviceSize)h * skin.half_size + skin.bound[h], count, memory))
        {
            VK_CALL(vkFreeMemory(skin.device->vk_device, memory, NULL));
            skin.retry_at = t0 + SKIN_RETRY_NS;
            failed = true;
            break;
        }
        skin.chunks[skin.chunk_count++] = memory;
        vkd3d_atomic_uint32_store_explicit(&skin.bound[h], skin.bound[h] + count, vkd3d_memory_order_release);
    }
    skin.grows++;
    skin.grow_ns += vkd3d_get_current_time_ns() - t0;
    vkd3d_frametrace_event("skin grow %s %.3f ms half %u %#x -> %#x, target %#x, highest %#x, system memory %u MiB",
            failed ? "fail" : "ok", (vkd3d_get_current_time_ns() - t0) / 1e6, h, bound0, skin.bound[h], target,
            skin.highest[h], skin.sysmem_bytes >> 20);
    INFO("skinning buffer: halves backed %u / %u MiB of %u, highest allocations %.1f / %.1f MiB, %.2f ms in %"PRIu64" grows.\n",
            skin.bound[0] >> 20, skin.bound[1] >> 20, skin.half_size >> 20, skin.highest[0] / 1048576.0,
            skin.highest[1] / 1048576.0, skin.grow_ns / 1e6, skin.grows);
}

static uint32_t skin_target(uint32_t end)
{
    uint64_t t = (uint64_t)end + skin_margin;

    t = (t + SKIN_CHUNK - 1) / SKIN_CHUNK * SKIN_CHUNK;
    return (uint32_t)min(t, (uint64_t)skin.half_size);
}

/* Bytes [0, end) of half h are about to be handed out: make sure they are backed. */
static void skin_ensure(unsigned int h, uint32_t end)
{
    uint32_t bound = vkd3d_atomic_uint32_load_explicit(&skin.bound[h], vkd3d_memory_order_acquire);

    if ((uint64_t)end + skin_margin / 2 <= bound && end <= skin.highest[h])
        return;
    pthread_mutex_lock(&skin_lock);
    if (skin.resource && !skin.broken)
    {
        skin.highest[h] = max(skin.highest[h], end);
        if ((uint64_t)end + skin_margin / 2 > skin.bound[h] &&
                (end > skin.bound[h] || vkd3d_get_current_time_ns() >= skin.retry_at))
            skin_grow(h, skin_target(end));
        if ((end > skin.bound[h]) != skin.in_loss[h])
        {
            skin.in_loss[h] = !skin.in_loss[h];
            vkd3d_frametrace_event("skin loss %s half %u end %#x bound %#x", skin.in_loss[h] ? "start" : "end",
                    h, end, skin.bound[h]);
        }
        if (end > skin.bound[h] && !(skin.losses++ % 100))
            ERR("skinning buffer: LOSS: half %u allocated up to %#x but only %#x bytes are backed (%"PRIu64" so far).\n",
                    h, end, skin.bound[h], skin.losses);
    }
    pthread_mutex_unlock(&skin_lock);
}

static int skin_half_of(const struct engine_linear *a)
{
    if (!skin.resource)
        return -1;
    return a == skin.half[0] ? 0 : a == skin.half[1] ? 1 : -1;
}

/* 0x2162ef0, same operations: size rounded up to the allocator's alignment, lock xadd; past the end
 * the allocator is marked full (xchg) and 0 returned. */
void *vkd3d_skinbuf_alloc(void *allocator, int32_t size)
{
    struct engine_linear *a = allocator;
    uint32_t align = a->align, sz, old, limit;
    int h;

    sz = ((uint32_t)size - 1 + align) & (0u - align);
    old = __atomic_fetch_add(&a->used, sz, __ATOMIC_SEQ_CST);
    limit = a->size;
    if (old + sz > limit)
    {
        __atomic_exchange_n(&a->used, limit, __ATOMIC_SEQ_CST);
        return NULL;
    }
    if ((h = skin_half_of(a)) >= 0)
    {
        __atomic_fetch_add(&skin.allocs, 1, __ATOMIC_RELAXED);
        skin_ensure(h, old + sz);
    }
    return a->base + old;
}

/* 0x2162f20, same operations: room for `align2` alignment on top, the slack handed back with a
 * compare-exchange if nothing was allocated after it. */
void *vkd3d_skinbuf_alloc_aligned(void *allocator, int32_t size, int32_t align2)
{
    struct engine_linear *a = allocator;
    uint32_t align = a->align, sz, pad, total, old, end, start, limit, expected;
    int h;

    sz = ((uint32_t)size - 1 + align) & (0u - align);
    pad = (uint32_t)align2 - align;
    total = pad + sz;
    old = __atomic_fetch_add(&a->used, total, __ATOMIC_SEQ_CST);
    end = old + total;
    limit = a->size;
    if (end > limit)
    {
        __atomic_exchange_n(&a->used, limit, __ATOMIC_SEQ_CST);
        return NULL;
    }
    start = (pad + old) & (0u - (uint32_t)align2);
    if (start + sz < end)
    {
        expected = end;
        __atomic_compare_exchange_n(&a->used, &expected, start + sz, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
    if ((h = skin_half_of(a)) >= 0)
    {
        __atomic_fetch_add(&skin.allocs, 1, __ATOMIC_RELAXED);
        skin_ensure(h, start + sz);
    }
    return a->base + start;
}

/* The engine calls 0x2162ef0 / 0x2162f20 from code built with whole-program optimisation, which may
 * keep values in registers those two leaf functions never touch (rcx, r10, r11, xmm0-5) across the
 * call. These entry points preserve every register but rax, like the originals do for those. */
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl vkd3d_skinbuf_alloc_stub\n"
    "vkd3d_skinbuf_alloc_stub:\n"
    "    push %rcx\n"
    "    push %rdx\n"
    "    push %r8\n"
    "    push %r9\n"
    "    push %r10\n"
    "    push %r11\n"
    "    sub $0x88, %rsp\n"
    "    movdqu %xmm0, 0x20(%rsp)\n"
    "    movdqu %xmm1, 0x30(%rsp)\n"
    "    movdqu %xmm2, 0x40(%rsp)\n"
    "    movdqu %xmm3, 0x50(%rsp)\n"
    "    movdqu %xmm4, 0x60(%rsp)\n"
    "    movdqu %xmm5, 0x70(%rsp)\n"
    "    call vkd3d_skinbuf_alloc\n"
    "    movdqu 0x20(%rsp), %xmm0\n"
    "    movdqu 0x30(%rsp), %xmm1\n"
    "    movdqu 0x40(%rsp), %xmm2\n"
    "    movdqu 0x50(%rsp), %xmm3\n"
    "    movdqu 0x60(%rsp), %xmm4\n"
    "    movdqu 0x70(%rsp), %xmm5\n"
    "    add $0x88, %rsp\n"
    "    pop %r11\n"
    "    pop %r10\n"
    "    pop %r9\n"
    "    pop %r8\n"
    "    pop %rdx\n"
    "    pop %rcx\n"
    "    ret\n"
    ".p2align 4\n"
    ".globl vkd3d_skinbuf_alloc_aligned_stub\n"
    "vkd3d_skinbuf_alloc_aligned_stub:\n"
    "    push %rcx\n"
    "    push %rdx\n"
    "    push %r8\n"
    "    push %r9\n"
    "    push %r10\n"
    "    push %r11\n"
    "    sub $0x88, %rsp\n"
    "    movdqu %xmm0, 0x20(%rsp)\n"
    "    movdqu %xmm1, 0x30(%rsp)\n"
    "    movdqu %xmm2, 0x40(%rsp)\n"
    "    movdqu %xmm3, 0x50(%rsp)\n"
    "    movdqu %xmm4, 0x60(%rsp)\n"
    "    movdqu %xmm5, 0x70(%rsp)\n"
    "    call vkd3d_skinbuf_alloc_aligned\n"
    "    movdqu 0x20(%rsp), %xmm0\n"
    "    movdqu 0x30(%rsp), %xmm1\n"
    "    movdqu 0x40(%rsp), %xmm2\n"
    "    movdqu 0x50(%rsp), %xmm3\n"
    "    movdqu 0x60(%rsp), %xmm4\n"
    "    movdqu 0x70(%rsp), %xmm5\n"
    "    add $0x88, %rsp\n"
    "    pop %r11\n"
    "    pop %r10\n"
    "    pop %r9\n"
    "    pop %r8\n"
    "    pop %rdx\n"
    "    pop %rcx\n"
    "    ret\n"
);

HRESULT vkd3d_skinbuf_create_placed(struct d3d12_device *device, struct d3d12_heap *heap,
        uint64_t heap_offset, const D3D12_RESOURCE_DESC1 *desc, D3D12_RESOURCE_STATES initial_state,
        struct d3d12_resource **resource)
{
    const struct vkd3d_vk_device_procs *vk_procs = &device->vk_procs;
    VkMemoryRequirements requirements;
    struct d3d12_resource *object;
    VkFenceCreateInfo fence_info;
    uint32_t type_bits;
    uint8_t *image;
    HRESULT hr;

    if (desc->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || heap_offset || desc->Width != heap->desc.SizeInBytes ||
            !device->internal_sparse_queue || !(image = skin_engine_image()))
    {
        ERR("skinning buffer: unexpected placement (dimension %u, offset %#"PRIx64", width %#"PRIx64") in the "
                "memoryless heap.\n", desc->Dimension, heap_offset, desc->Width);
        return E_INVALIDARG;
    }

    if (FAILED(hr = d3d12_resource_create_reserved(device, desc, initial_state, NULL, 0, NULL, &object)))
        return hr;
    if (!(object->flags & VKD3D_RESOURCE_RESERVED))
    {
        ERR("skinning buffer: reserved buffer fell back to a committed one.\n");
        *resource = object;
        return S_OK;
    }
    d3d12_resource_wait_for_sparse_init(object);

    VK_CALL(vkGetBufferMemoryRequirements(device->vk_device, object->res.vk_buffer, &requirements));
    type_bits = requirements.memoryTypeBits;
    for (skin.memory_type = 0; skin.memory_type < device->memory_properties.memoryTypeCount; skin.memory_type++)
    {
        VkMemoryPropertyFlags flags = device->memory_properties.memoryTypes[skin.memory_type].propertyFlags;

        if ((type_bits & (1u << skin.memory_type)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            break;
    }
    skin.type_bits = type_bits;
    skin.fallback_type = device->memory_properties.memoryTypeCount;

    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    pthread_mutex_lock(&skin_lock);
    skin.device = device;
    skin.half_size = (uint32_t)(desc->Width / 2);
    skin.half[0] = (struct engine_linear *)(image + SKIN_MGR_RVA + 0x10);
    skin.half[1] = (struct engine_linear *)(image + SKIN_MGR_RVA + 0x90);
    skin.bound[0] = skin.bound[1] = 0;
    if (skin.memory_type == device->memory_properties.memoryTypeCount ||
            VK_CALL(vkCreateFence(device->vk_device, &fence_info, NULL, &skin.fence)) < 0)
    {
        ERR("skinning buffer: setup failed (memory types %#x); the buffer stays unbacked.\n", type_bits);
        skin.broken = true;
    }
    else
    {
        /* The halves are set up by the manager right after this returns (used = 0). */
        skin.resource = object;
        skin_grow(0, skin_target(skin_initial > skin_margin ? skin_initial - skin_margin : 0));
        skin_grow(1, skin_target(skin_initial > skin_margin ? skin_initial - skin_margin : 0));
    }
    pthread_mutex_unlock(&skin_lock);

    INFO("skinning buffer: sparse, %u MiB, VA %#"PRIx64", memory type %u (types %#x), initial %u MiB "
            "per half, margin %u MiB.\n", (unsigned int)(desc->Width >> 20), object->res.va, skin.memory_type,
            skin.type_bits, skin_initial >> 20, skin_margin >> 20);
    *resource = object;
    return S_OK;
}

/* Every ExecuteCommandLists: a half's `used` past its bound means an allocation the hooks did not
 * see (or growth that failed); count it, and grow. */
void vkd3d_skinbuf_update(void)
{
    unsigned int h;
    uint32_t used;

    if (!skin.resource || skin.broken)
        return;
    for (h = 0; h < 2; h++)
    {
        used = __atomic_load_n(&skin.half[h]->used, __ATOMIC_RELAXED);
        if (used <= vkd3d_atomic_uint32_load_explicit(&skin.bound[h], vkd3d_memory_order_acquire))
            continue;
        pthread_mutex_lock(&skin_lock);
        if (skin.resource && used > skin.bound[h])
        {
            if (!(skin.unbacked_checks++ % 100))
                ERR("skinning buffer: LOSS: half %u has %#x bytes allocated, %#x backed (%"PRIu64" submissions so far).\n",
                        h, used, skin.bound[h], skin.unbacked_checks);
            skin.highest[h] = max(skin.highest[h], used);
            if (vkd3d_get_current_time_ns() >= skin.retry_at)
                skin_grow(h, skin_target(used));
        }
        pthread_mutex_unlock(&skin_lock);
    }
}

/* VRAM the game should leave free: two margins of growth, plus any bytes still wanted. */
uint64_t vkd3d_skinbuf_budget_reserve(void)
{
    uint64_t r;
    unsigned int h;

    if (!skin.resource || skin.broken)
        return 0;
    r = 2ull * skin_margin;
    for (h = 0; h < 2; h++)
    {
        uint32_t wanted = vkd3d_atomic_uint32_load_explicit(&skin.wanted[h], vkd3d_memory_order_relaxed);
        uint32_t bound = vkd3d_atomic_uint32_load_explicit(&skin.bound[h], vkd3d_memory_order_relaxed);

        r += wanted > bound ? wanted - bound : 0;
    }
    return r;
}

void vkd3d_skinbuf_resource_destroyed(struct d3d12_resource *resource)
{
    const struct vkd3d_vk_device_procs *vk_procs;
    size_t i;

    if (!resource || resource != skin.resource)
        return;
    pthread_mutex_lock(&skin_lock);
    vk_procs = &skin.device->vk_procs;
    INFO("skinning buffer: destroyed; backed %u / %u MiB (%u MiB in system memory), highest allocations %u / %u bytes, "
            "%"PRIu64" allocations, %"PRIu64" losses, %"PRIu64" submissions over a bound, %"PRIu64" failed VRAM "
            "allocations.\n", skin.bound[0] >> 20, skin.bound[1] >> 20, skin.sysmem_bytes >> 20, skin.highest[0],
            skin.highest[1], skin.allocs, skin.losses, skin.unbacked_checks, skin.failed_allocs);
    for (i = 0; i < skin.chunk_count; i++)
        VK_CALL(vkFreeMemory(skin.device->vk_device, skin.chunks[i], NULL));
    if (skin.fence)
        VK_CALL(vkDestroyFence(skin.device->vk_device, skin.fence, NULL));
    vkd3d_free(skin.chunks);
    memset(&skin, 0, sizeof(skin));
    pthread_mutex_unlock(&skin_lock);
}

/* Periodic summary for the bench logs (from the frame trace's once-a-second stats). */
void vkd3d_skinbuf_log(void)
{
    static uint64_t last;
    uint64_t now;

    if (!skin.resource || (now = vkd3d_get_current_time_ns()) - last < 30000000000ull)
        return;
    last = now;
    INFO("skinning buffer: backed %u / %u MiB of %u, highest allocations %.1f / %.1f MiB, %"PRIu64" allocations, "
            "%"PRIu64" losses, %"PRIu64" submissions over a bound.\n", skin.bound[0] >> 20, skin.bound[1] >> 20,
            skin.half_size >> 20, skin.highest[0] / 1048576.0, skin.highest[1] / 1048576.0, skin.allocs, skin.losses,
            skin.unbacked_checks);
}
