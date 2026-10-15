/*
 * Bench instrumentation (miles-bench): when VKD3D_MEMTRACE names a file, log the lifetime of
 * every D3D12 resource and heap plus every VkDeviceMemory allocation as tab-separated lines.
 * Each creation records the first stack frame outside d3d12core.dll (module!+rva), so an
 * allocation can be traced back to the function in the game binary that asked for it.
 *
 * Line formats (t = milliseconds since the trace was opened):
 *   MT  type_index  property_flags  heap_index
 *   MH  heap_index  size  heap_flags
 *   RC  t ptr kind dim width height depth mips format flags layout heap_type heap_flags
 *       heap_ptr heap_offset req_size vk_type suballocated exe_rva callers
 *   RD  t ptr
 *   HC  t ptr size heap_type cpu_page pool flags vk_type suballocated exe_rva callers
 *   HD  t ptr
 *   N   t ptr name
 *   VA  t vk_type size ok budgeted vk_memory
 *   VF  t vk_type size vk_memory
 *   DC  t ptr type num_descriptors flags descriptor_buffer_bytes vk_type aux_bytes aux_vk_type exe_rva callers
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"
#include "vkd3d_utf8.h"

#include <stdio.h>

static pthread_once_t memtrace_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t memtrace_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *memtrace_file;
static uint64_t memtrace_t0;
static uint64_t memtrace_last_flush;
static unsigned int memtrace_unflushed;
static bool memtrace_props_logged;

static void memtrace_open(void)
{
    char path[VKD3D_PATH_MAX];

    if (!vkd3d_get_env_var("VKD3D_MEMTRACE", path, sizeof(path)) || !*path)
        return;

    if (!(memtrace_file = fopen(path, "w")))
    {
        ERR("memtrace: failed to open %s.\n", path);
        return;
    }

    setvbuf(memtrace_file, NULL, _IOFBF, 1 << 20);
    memtrace_t0 = vkd3d_get_current_time_ns();
    fprintf(memtrace_file, "# memtrace v1\n");
    INFO("memtrace: logging to %s.\n", path);
}

static bool memtrace_enabled(void)
{
    pthread_once(&memtrace_once, memtrace_open);
    return memtrace_file != NULL;
}

static double memtrace_now_ms(void)
{
    return (vkd3d_get_current_time_ns() - memtrace_t0) / 1e6;
}

/* Caller must hold memtrace_lock. */
static void memtrace_end_line(void)
{
    uint64_t now = vkd3d_get_current_time_ns();

    if (++memtrace_unflushed >= 256 || now - memtrace_last_flush > 1000000000ull)
    {
        fflush(memtrace_file);
        memtrace_unflushed = 0;
        memtrace_last_flush = now;
    }
}

/* Caller must hold memtrace_lock. */
static void memtrace_log_memory_properties(struct d3d12_device *device)
{
    const VkPhysicalDeviceMemoryProperties *props = &device->memory_properties;
    uint32_t i;

    if (memtrace_props_logged)
        return;
    memtrace_props_logged = true;

    for (i = 0; i < props->memoryHeapCount; i++)
        fprintf(memtrace_file, "MH\t%u\t%"PRIu64"\t%#x\n", i, props->memoryHeaps[i].size, props->memoryHeaps[i].flags);
    for (i = 0; i < props->memoryTypeCount; i++)
        fprintf(memtrace_file, "MT\t%u\t%#x\t%u\n", i, props->memoryTypes[i].propertyFlags, props->memoryTypes[i].heapIndex);
}

#ifdef _WIN32
struct memtrace_module
{
    HMODULE handle;
    uintptr_t end;
    char name[64];
};

static struct memtrace_module memtrace_modules[32];
static unsigned int memtrace_module_count;
static HMODULE memtrace_self, memtrace_exe;

/* Caller must hold memtrace_lock. */
static const struct memtrace_module *memtrace_lookup_module(const void *address)
{
    IMAGE_NT_HEADERS *nt;
    char path[MAX_PATH];
    const char *base;
    HMODULE handle;
    unsigned int i;

    for (i = 0; i < memtrace_module_count; i++)
    {
        if ((uintptr_t)address >= (uintptr_t)memtrace_modules[i].handle && (uintptr_t)address < memtrace_modules[i].end)
            return &memtrace_modules[i];
    }

    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address, &handle))
        return NULL;

    if (memtrace_module_count == ARRAY_SIZE(memtrace_modules))
        return NULL;

    nt = (IMAGE_NT_HEADERS *)((char *)handle + ((IMAGE_DOS_HEADER *)handle)->e_lfanew);
    memtrace_modules[i].handle = handle;
    memtrace_modules[i].end = (uintptr_t)handle + nt->OptionalHeader.SizeOfImage;
    GetModuleFileNameA(handle, path, sizeof(path));
    base = strrchr(path, '\\');
    snprintf(memtrace_modules[i].name, sizeof(memtrace_modules[i].name), "%s", base ? base + 1 : path);
    memtrace_module_count++;
    return &memtrace_modules[i];
}

/* Writes "\texe_rva\tcallers". Caller must hold memtrace_lock. */
static void memtrace_write_callers(void)
{
    const struct memtrace_module *module;
    unsigned int i, count, written = 0;
    static unsigned int depth;
    uintptr_t exe_rva = 0;
    void *frames[48];
    char chain[1024];
    size_t len = 0;
    char env[8];

    if (!depth)
        depth = vkd3d_get_env_var("VKD3D_MEMTRACE_DEPTH", env, sizeof(env)) && atoi(env) > 0 ? min(atoi(env), 40) : 4;

    if (!memtrace_self)
    {
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                (const void *)memtrace_write_callers, &memtrace_self);
        memtrace_exe = GetModuleHandleA(NULL);
    }

    chain[0] = '\0';
    count = RtlCaptureStackBackTrace(1, ARRAY_SIZE(frames), frames, NULL);

    for (i = 0; i < count; i++)
    {
        if (!(module = memtrace_lookup_module(frames[i])) || module->handle == memtrace_self)
            continue;

        if (module->handle == memtrace_exe && !exe_rva)
            exe_rva = (uintptr_t)frames[i] - (uintptr_t)module->handle;

        if (written < depth && len < sizeof(chain))
        {
            len += snprintf(chain + len, sizeof(chain) - len, "%s%s!+%#"PRIxPTR,
                    written ? "<" : "", module->name, (uintptr_t)frames[i] - (uintptr_t)module->handle);
            written++;
        }

        if (exe_rva && written >= depth)
            break;
    }

    fprintf(memtrace_file, "\t%#"PRIxPTR"\t%s", exe_rva, chain);
}
#else
static void memtrace_write_callers(void)
{
    fprintf(memtrace_file, "\t0\t");
}
#endif

void vkd3d_memtrace_resource_create(struct d3d12_device *device, struct d3d12_resource *resource, const char *kind)
{
    const struct vkd3d_vk_device_procs *vk_procs = &device->vk_procs;
    const D3D12_RESOURCE_DESC1 *desc = &resource->desc;
    const struct vkd3d_memory_allocation *mem;
    VkMemoryRequirements requirements;
    uint64_t req_size, heap_offset = 0;
    D3D12_HEAP_FLAGS heap_flags;
    D3D12_HEAP_TYPE heap_type;

    if (!memtrace_enabled())
        return;

    if (d3d12_resource_is_texture(resource) && resource->res.vk_image)
    {
        VK_CALL(vkGetImageMemoryRequirements(device->vk_device, resource->res.vk_image, &requirements));
        req_size = requirements.size;
    }
    else
        req_size = desc->Width;

    if (resource->heap)
    {
        mem = &resource->heap->allocation;
        heap_type = resource->heap->desc.Properties.Type;
        heap_flags = resource->heap->desc.Flags;
        heap_offset = resource->mem.offset - mem->offset;
    }
    else
    {
        mem = &resource->mem;
        heap_type = resource->heap_properties.Type;
        heap_flags = resource->heap_flags;
    }

    pthread_mutex_lock(&memtrace_lock);
    memtrace_log_memory_properties(device);
    fprintf(memtrace_file, "RC\t%.3f\t%p\t%s\t%u\t%"PRIu64"\t%u\t%u\t%u\t%u\t%#x\t%u\t%u\t%#x\t%p\t%"PRIu64"\t%"PRIu64"\t%u\t%u",
            memtrace_now_ms(), (void *)resource, kind, desc->Dimension, desc->Width, desc->Height,
            desc->DepthOrArraySize, desc->MipLevels, desc->Format, desc->Flags, desc->Layout,
            heap_type, heap_flags, (void *)resource->heap, heap_offset, req_size,
            mem->device_allocation.vk_memory_type, mem->chunk != NULL);
    memtrace_write_callers();
    fputc('\n', memtrace_file);
    /* RK resource cookie: joins descriptor QA access counts (VKD3D_BENCH_QACOUNT) to resources. */
    fprintf(memtrace_file, "RK\t%.3f\t%p\t%u\n", memtrace_now_ms(), (void *)resource, resource->res.cookie.index);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}

void vkd3d_memtrace_resource_destroy(struct d3d12_resource *resource)
{
    if (!memtrace_enabled())
        return;

    pthread_mutex_lock(&memtrace_lock);
    fprintf(memtrace_file, "RD\t%.3f\t%p\n", memtrace_now_ms(), (void *)resource);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}

void vkd3d_memtrace_heap_create(struct d3d12_device *device, struct d3d12_heap *heap)
{
    const D3D12_HEAP_DESC *desc = &heap->desc;

    if (!memtrace_enabled())
        return;

    pthread_mutex_lock(&memtrace_lock);
    memtrace_log_memory_properties(device);
    fprintf(memtrace_file, "HC\t%.3f\t%p\t%"PRIu64"\t%u\t%u\t%u\t%#x\t%u\t%u",
            memtrace_now_ms(), (void *)heap, desc->SizeInBytes, desc->Properties.Type,
            desc->Properties.CPUPageProperty, desc->Properties.MemoryPoolPreference, desc->Flags,
            heap->allocation.device_allocation.vk_memory_type, heap->allocation.chunk != NULL);
    memtrace_write_callers();
    fputc('\n', memtrace_file);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}

void vkd3d_memtrace_descriptor_heap_create(struct d3d12_device *device, struct d3d12_descriptor_heap *heap)
{
    if (!memtrace_enabled())
        return;

    pthread_mutex_lock(&memtrace_lock);
    memtrace_log_memory_properties(device);
    fprintf(memtrace_file, "DC\t%.3f\t%p\t%u\t%u\t%#x\t%"PRIu64"\t%u\t%"PRIu64"\t%u",
            memtrace_now_ms(), (void *)heap, heap->desc.Type, heap->desc.NumDescriptors, heap->desc.Flags,
            (uint64_t)heap->descriptor_buffer.device_allocation.size,
            heap->descriptor_buffer.device_allocation.vk_memory_type,
            (uint64_t)heap->device_allocation.size, heap->device_allocation.vk_memory_type);
    memtrace_write_callers();
    fputc('\n', memtrace_file);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}

void vkd3d_memtrace_heap_destroy(struct d3d12_heap *heap)
{
    if (!memtrace_enabled())
        return;

    pthread_mutex_lock(&memtrace_lock);
    fprintf(memtrace_file, "HD\t%.3f\t%p\n", memtrace_now_ms(), (void *)heap);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}

void vkd3d_memtrace_set_name(const void *object, const WCHAR *name)
{
    char *utf8;

    if (!memtrace_enabled() || !name)
        return;

    if (!(utf8 = vkd3d_strdup_w_utf8(name, 0)))
        return;

    pthread_mutex_lock(&memtrace_lock);
    fprintf(memtrace_file, "N\t%.3f\t%p\t%s\n", memtrace_now_ms(), object, utf8);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
    vkd3d_free(utf8);
}

void vkd3d_memtrace_vk_allocate(struct d3d12_device *device, uint32_t vk_type, uint64_t size,
        bool ok, bool budgeted, VkDeviceMemory vk_memory)
{
    if (!memtrace_enabled())
        return;

    pthread_mutex_lock(&memtrace_lock);
    memtrace_log_memory_properties(device);
    fprintf(memtrace_file, "VA\t%.3f\t%u\t%"PRIu64"\t%u\t%u\t%#"PRIx64"\n",
            memtrace_now_ms(), vk_type, size, ok, budgeted, (uint64_t)vk_memory);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}

static pthread_once_t l0_sysmem_once = PTHREAD_ONCE_INIT;
static bool l0_sysmem_enabled;

static void l0_sysmem_init(void)
{
    char value[8];

    l0_sysmem_enabled = vkd3d_get_env_var("VKD3D_BENCH_L0_SYSMEM", value, sizeof(value)) && value[0] == '1';
    if (l0_sysmem_enabled)
        INFO("bench: CUSTOM heaps with MEMORY_POOL_L0 and no CPU access go to system memory.\n");
}

/* With VKD3D_BENCH_L0_SYSMEM=1, CUSTOM heaps asking for MEMORY_POOL_L0 without CPU access are placed in
 * system memory on discrete GPUs, as D3D12 does on Windows. Stock vkd3d-proton puts them in VRAM, so a game
 * that demotes resources out of VRAM this way (Nixxes ports) sees no drop in VRAM usage and keeps demoting. */
bool vkd3d_bench_l0_sysmem(struct d3d12_device *device, const D3D12_HEAP_PROPERTIES *heap_properties)
{
    if (heap_properties->Type != D3D12_HEAP_TYPE_CUSTOM ||
            heap_properties->MemoryPoolPreference != D3D12_MEMORY_POOL_L0 ||
            heap_properties->CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE)
        return false;

    pthread_once(&l0_sysmem_once, l0_sysmem_init);
    return l0_sysmem_enabled && !d3d12_device_is_uma(device, NULL);
}

/* -1 until the first device: whether the GPU has VRAM of its own (VkPhysicalDeviceType DISCRETE). On
 * an integrated GPU "VRAM" is system memory: nothing to keep free, nothing to cache. */
static int bench_gpu_discrete = -1;

void vkd3d_bench_set_gpu_type(VkPhysicalDeviceType type)
{
    if (bench_gpu_discrete < 0)
        bench_gpu_discrete = type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

bool vkd3d_bench_gpu_integrated(void)
{
    return bench_gpu_discrete == 0;
}

static pthread_once_t vram_reserve_once = PTHREAD_ONCE_INIT;
static bool vram_reserve_enabled;
static VkDeviceSize vram_reserve_bytes;
static bool vram_reserve_texture_heaps;
static uint64_t vram_refused_count, vram_refused_bytes;

/* The reserve's size: VKD3D_BENCH_VRAM_RESERVE_MB=N sets it; with VKD3D_BENCH_VRAM_RESERVE=1 it is
 * what earlier sessions of this build measured (vkd3d_bench_vram_hot below), and before anything was
 * learned there is none, as without the patch, while this session measures it. */
static void vram_reserve_init(void)
{
    const char *source = NULL;
    uint64_t learned;
    char value[64];

    if (vkd3d_get_env_var("VKD3D_BENCH_VRAM_RESERVE_MB", value, sizeof(value)) && *value)
    {
        vram_reserve_bytes = strtoull(value, NULL, 10) << 20;
        source = "VKD3D_BENCH_VRAM_RESERVE_MB";
    }
    else if (vkd3d_get_env_var("VKD3D_BENCH_VRAM_RESERVE", value, sizeof(value)) && value[0] == '1')
    {
        if (vkd3d_bench_learned_get(VKD3D_BENCH_LEARNED_RESERVE_BYTES, &learned))
        {
            vram_reserve_bytes = learned;
            source = "learned";
        }
        else
            source = "nothing learned yet; measuring";
    }
    vram_reserve_enabled = !!source;
    if (vram_reserve_enabled && vkd3d_bench_gpu_integrated())
    {
        INFO("bench: not a discrete GPU; no VRAM reserve.\n");
        vram_reserve_enabled = false;
    }
    /* Game texture heaps (DEFAULT, texture-only, no RT/DS) mostly hold streamed textures but also a few
     * small UAV textures, so they are only treated as cold on request. */
    vram_reserve_texture_heaps = vkd3d_get_env_var("VKD3D_BENCH_VRAM_RESERVE_TEXHEAPS", value, sizeof(value)) &&
            value[0] == '1';
    if (vram_reserve_enabled)
    {
        INFO("bench: keeping the last %"PRIu64" MiB of VRAM for render targets, depth and UAVs (%s)%s.\n",
                vram_reserve_bytes >> 20, source, vram_reserve_texture_heaps ? " (texture heaps are cold too)" : "");
    }
}

/* With the reserve on, allocations that are only ever sampled stop taking VRAM once less than the
 * reserve is left of the budget, and fall back to system memory instead. Placement in vkd3d-proton is
 * fixed at allocation time (no WDDM-style paging), so without this, render targets created after VRAM
 * fills up (shadow cascades and full-res depth at level load, in Miles Morales) live in system memory
 * for the rest of the session and the GPU writes and reads them over PCIe every frame. Returns the
 * reserve that applies to this allocation, 0 if it may use all of VRAM. *vram_class: COLD for what the
 * reserve holds back, HOT for what it is for (render targets, depth, UAV textures, buffers), OTHER for
 * neither (the game's texture heaps, mostly streamed textures but with a few UAV textures, which are
 * cold only with VKD3D_BENCH_VRAM_RESERVE_TEXHEAPS=1). */
VkDeviceSize vkd3d_bench_vram_reserve(const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags,
        uint32_t allocation_flags, enum vkd3d_bench_vram_class *vram_class)
{
    pthread_once(&vram_reserve_once, vram_reserve_init);
    *vram_class = VKD3D_BENCH_VRAM_OTHER;
    if (!vram_reserve_enabled)
        return 0;

    /* Buffers (UAV and constant buffers) and anything that can hold render targets, depth or UAVs stay hot. */
    if ((allocation_flags & VKD3D_ALLOCATION_FLAG_BENCH_HOT) || !(heap_flags & D3D12_HEAP_FLAG_DENY_BUFFERS) ||
            !(heap_flags & D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES))
    {
        *vram_class = VKD3D_BENCH_VRAM_HOT;
        return 0;
    }

    /* The engine's demotion pool: CUSTOM, L0, no CPU access. */
    if (heap_properties->Type == D3D12_HEAP_TYPE_CUSTOM)
    {
        if (heap_properties->MemoryPoolPreference == D3D12_MEMORY_POOL_L0)
            *vram_class = VKD3D_BENCH_VRAM_COLD;
    }
    else if ((allocation_flags & VKD3D_ALLOCATION_FLAG_BENCH_COMMITTED) || vram_reserve_texture_heaps)
        *vram_class = VKD3D_BENCH_VRAM_COLD;
    return *vram_class == VKD3D_BENCH_VRAM_COLD ? vram_reserve_bytes : 0;
}

/* Learning the reserve. VRAM first runs short when a cold allocation is refused or an allocation of
 * VRAM fails. From then on, everything that is not cold (render targets, depth, UAVs, buffers, the
 * game's texture heaps, vkd3d's own chunks) eats into the reserve, so the reserve this session
 * needed is its size minus the least VRAM left free after any allocation of VRAM, where an allocation
 * that failed (it fell back to system memory) counts as that much less than nothing
 * (vkd3d_bench_vram_headroom). The most any session of this build needed is learned (learned.c) and
 * kept back by later sessions. With VRAM to spare nothing runs short and nothing is learned.
 * vkd3d_bench_vram_hot keeps the hot share of it for the log. */
static bool vram_short;
static int64_t vram_min_free = INT64_MAX, vram_largest_late;
static pthread_mutex_t vram_late_lock = PTHREAD_MUTEX_INITIALIZER;
static struct vram_late { VkDeviceMemory memory; VkDeviceSize size; } *vram_late;
static size_t vram_late_count, vram_late_size;
static uint64_t vram_late_bytes, vram_late_peak, vram_late_logged;

void vkd3d_bench_vram_short(void)
{
    pthread_once(&vram_reserve_once, vram_reserve_init);
    if (vram_reserve_enabled && !__atomic_exchange_n(&vram_short, true, __ATOMIC_ACQ_REL))
        INFO("bench: VRAM ran short; hot allocations from here on are what the reserve has to make room for.\n");
}

bool vkd3d_bench_vram_learning(void)
{
    return vram_reserve_enabled && __atomic_load_n(&vram_short, __ATOMIC_ACQUIRE);
}

/* After an allocation of VRAM that is not cold, while learning: `free` is the VRAM it left (budget
 * minus usage). One that failed was short by at least its size, whatever the budget said (the driver
 * refused 6 MiB allocations with ~20 MiB of budget left, v15-S13-1). */
void vkd3d_bench_vram_headroom(int64_t free, VkDeviceSize size, bool failed)
{
    int64_t needed;

    if (!vkd3d_bench_vram_learning())
        return;
    pthread_mutex_lock(&vram_late_lock);
    vram_largest_late = max(vram_largest_late, (int64_t)size);
    /* A failed allocation: the game's streamer took too much; it makes room from its next budget query. */
    if (failed)
    {
        free = min(free, 0) - (int64_t)size;
        vkd3d_bench_hold_raise(size);
    }
    if (free < vram_min_free)
    {
        vram_min_free = free;
        needed = (int64_t)vram_reserve_bytes - free;
        if (needed > 0)
            vkd3d_bench_learned_raise(VKD3D_BENCH_LEARNED_RESERVE_BYTES, (uint64_t)needed);
        INFO("bench: %.1f MiB of VRAM left after a %.1f MiB allocation: this session needed a %.1f MiB reserve "
                "(largest late allocation %.1f MiB).\n", free / 1048576.0, size / 1048576.0,
                max(needed, 0) / 1048576.0, vram_largest_late / 1048576.0);
    }
    pthread_mutex_unlock(&vram_late_lock);
}

void vkd3d_bench_vram_hot(VkDeviceMemory memory, VkDeviceSize size)
{
    pthread_once(&vram_reserve_once, vram_reserve_init);
    if (!vram_reserve_enabled || !__atomic_load_n(&vram_short, __ATOMIC_ACQUIRE) || memory == VK_NULL_HANDLE)
        return;
    pthread_mutex_lock(&vram_late_lock);
    if (vkd3d_array_reserve((void **)&vram_late, &vram_late_size, vram_late_count + 1, sizeof(*vram_late)))
    {
        vram_late[vram_late_count].memory = memory;
        vram_late[vram_late_count].size = size;
        __atomic_store_n(&vram_late_count, vram_late_count + 1, __ATOMIC_RELEASE);
        vram_late_bytes += size;
        if (vram_late_bytes > vram_late_peak)
        {
            vram_late_peak = vram_late_bytes;
            if (vram_late_peak >= vram_late_logged + (16u << 20))
            {
                vram_late_logged = vram_late_peak;
                INFO("bench: %"PRIu64" MiB of hot memory asked for since VRAM ran short (reserve %"PRIu64" MiB).\n",
                        vram_late_peak >> 20, vram_reserve_bytes >> 20);
            }
        }
    }
    pthread_mutex_unlock(&vram_late_lock);
}

void vkd3d_bench_vram_freed(VkDeviceMemory memory)
{
    size_t i;

    if (!__atomic_load_n(&vram_late_count, __ATOMIC_ACQUIRE))
        return;
    pthread_mutex_lock(&vram_late_lock);
    for (i = 0; i < vram_late_count; i++)
    {
        if (vram_late[i].memory == memory)
        {
            vram_late_bytes -= vram_late[i].size;
            vram_late[i] = vram_late[vram_late_count - 1];
            __atomic_store_n(&vram_late_count, vram_late_count - 1, __ATOMIC_RELEASE);
            break;
        }
    }
    pthread_mutex_unlock(&vram_late_lock);
}

/* Allocate size bytes of system memory (a type out of type_bits without DEVICE_LOCAL) for the lazy
 * light grid and skinning buffer once VRAM is full. Types that are not host-visible are tried
 * first, then the host-visible ones; the first allocation that succeeds wins. Not every such type
 * takes plain allocations (in the game's device, type 11, which the GPU itself does not list,
 * refuses them), so each is tried. Returns the type used, or memoryTypeCount and VK_NULL_HANDLE. */
uint32_t vkd3d_bench_sysmem_alloc(struct d3d12_device *device, uint32_t type_bits, VkDeviceSize size,
        VkDeviceMemory *memory)
{
    const VkPhysicalDeviceMemoryProperties *props = &device->memory_properties;
    const struct vkd3d_vk_device_procs *vk_procs = &device->vk_procs;
    VkMemoryAllocateInfo info;
    unsigned int pass;
    uint32_t i;

    memset(&info, 0, sizeof(info));
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = size;
    for (pass = 0; pass < 2; pass++)
    {
        for (i = 0; i < props->memoryTypeCount; i++)
        {
            VkMemoryPropertyFlags flags = props->memoryTypes[i].propertyFlags;

            if (!(type_bits & (1u << i)) || (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
                    !!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != pass)
                continue;
            info.memoryTypeIndex = i;
            if (VK_CALL(vkAllocateMemory(device->vk_device, &info, NULL, memory)) >= 0)
                return i;
        }
    }
    *memory = VK_NULL_HANDLE;
    return props->memoryTypeCount;
}

bool vkd3d_bench_vram_reserve_chunks(void)
{
    static int enabled = -1;
    char value[16];

    if (enabled < 0)
        enabled = vkd3d_get_env_var("VKD3D_BENCH_VRAM_RESERVE_CHUNKS", value, sizeof(value)) && value[0] == '1';
    return enabled;
}

void vkd3d_bench_vram_refused(uint64_t size)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    uint64_t count, bytes;

    pthread_mutex_lock(&lock);
    count = ++vram_refused_count;
    bytes = vram_refused_bytes += size;
    pthread_mutex_unlock(&lock);

    if (!(count & (count - 1)))
        INFO("bench: %"PRIu64" cold allocations (%"PRIu64" MiB) sent to system memory to keep the VRAM reserve.\n",
                count, bytes >> 20);
}

void vkd3d_memtrace_vk_free(uint32_t vk_type, uint64_t size, VkDeviceMemory vk_memory)
{
    if (!memtrace_enabled())
        return;

    pthread_mutex_lock(&memtrace_lock);
    fprintf(memtrace_file, "VF\t%.3f\t%u\t%"PRIu64"\t%#"PRIx64"\n",
            memtrace_now_ms(), vk_type, size, (uint64_t)vk_memory);
    memtrace_end_line();
    pthread_mutex_unlock(&memtrace_lock);
}
