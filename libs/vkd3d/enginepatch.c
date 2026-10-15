/*
 * Engine patches for MilesMorales.exe (miles-bench), applied in memory at the first device
 * creation when VKD3D_ENGINE_PATCH lists them. Every patch site is checked byte for byte
 * against the expected build (PE timestamp + SizeOfImage), and a mismatch skips the site.
 *
 *   spin  Every CPU wait for the GPU in the renderer (PushBuffer::WaitForFrame,
 *         RenderPipeline::WaitFor, RenderDataRing::WaitForTail, ...) is a busy loop:
 *             while (fence->GetCompletedValue() < target) { 50 x pause; check timeout; }
 *         which keeps one of the four cores at 100% while the GPU works. The patch redirects
 *         the pause call at each of those sites to a hook that sleeps until the fence polled
 *         just before it advances, then returns to the engine's loop, which re-checks its own
 *         condition. Synchronisation is unchanged, so rendering is identical. The same goes for
 *         the query readback loop, whose GetData polls the query's fence internally. A poll
 *         older than ~100us doesn't count (the loop iteration didn't reach the fence), and the
 *         hook then falls back to the original pause.
 *
 *   qdefer  FinalizeQueriesBlocking (render command 0x75, handler 0x1d98700) flushes the command
 *         list and then spins until the GPU has answered every query in its batch: occlusion
 *         pixel counts that the game reads back for lens flares/coronas and visibility checks.
 *         The queries were drawn just before (ProcessQueries), so the render thread waits for
 *         the GPU to drain its whole queue, twice a frame, and the GPU then idles until the
 *         thread records more work. The patch takes every answer that is already there and
 *         defers the rest to the end of the render thread's frame (its call to
 *         PushBuffer::WaitForFrame), where it waits for them the same way. Results reach the
 *         query objects before the render thread hands the frame back to the game thread, in the
 *         same values; only the mid-frame stall goes. Q lines report how often it deferred and
 *         how long the end of the frame then had to wait.
 *
 * VKD3D_FRAMETRACE=<file> logs one line per Present for frame-time analysis:
 *   F  t_ms  wait_ms  draws  wait_ms = time the presenting thread spent in GPU waits since the
 *                            previous present (blocked, or spinning when the hook is off);
 *                            draws = draw, dispatch and ExecuteIndirect calls on any thread
 *   M  t_ms  count  name     PIX event names seen during that frame, dumped for frames with
 *                            more than 1.6x the average draw count and for every 500th frame
 *   R  t_ms  draws  WxHxD  format  layer  rt|ds  name
 *                            draws per render target (first RTV, else the DSV) in those frames
 *   S  t_ms  waits  blocked_ms  timeouts  spin_ms   once a second, process totals
 *
 * The control block (engine_ctl) starts with a magic string so bench/enginectl.py can find
 * it through /proc/<pid>/mem and flip spin_block at runtime for same-session A/B runs.
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#include <stdarg.h>
#include <stdio.h>


#define ENGINE_EXE_TIMESTAMP 0x6374c7f4u
#define ENGINE_EXE_SIZE 0x8fef000u
#define ENGINE_PAUSE_HELPER 0x20c6840u
#define ENGINE_SPIN_TIMEOUT_MS 20
#define ENGINE_POLL_FRESH_TSC 400000 /* ~100us at 3.4-4 GHz */

struct engine_ctl
{
    char magic[16];
    volatile uint32_t spin_block;
    volatile uint32_t sites_patched;
    volatile uint64_t waits;
    volatile uint64_t blocked_ns;
    volatile uint64_t timeouts;
    volatile uint64_t spin_ns;
    /* Live switches for same-session A/B (bench/abf.py): ENGINE_FEATURE_* */
    volatile uint32_t features;
};

struct engine_ctl engine_ctl = {"MILESBENCH_CTL1", 1};

uint32_t vkd3d_bench_features(void)
{
    return engine_ctl.features;
}

/* Per patched site: time the engine spent waiting there (blocked or spinning), for W lines. */
static uint8_t *engine_image;
static uint64_t engine_site_ns[16], engine_site_count[16];

/* Call sites of the 50 x pause helper inside loops that poll ID3D12Fence::GetCompletedValue. */
static const struct
{
    uint32_t rva;
    const char *name;
}
engine_spin_sites[] =
{
    {0x1d71755, "fence wait (!= value)"},
    {0x1d7c6ab, "PushBuffer::WaitForFrame"},
    {0x1d7fa50, "RenderDataRing::Shutdown"},
    {0x1d7fc7f, "RenderDataRing::WaitForTail"},
    {0x1d95cc9, "RenderPipeline::WaitFor"},
    {0x1d95ddf, "RenderPipeline::WaitFor"},
    {0x1d95ee8, "RenderPipeline::WaitFor"},
    {0x1d96279, "RenderPipeline::WaitFor"},
    {0x1d98783, "query readback (GetData)"},
    {0x1df9e50, "fence wait (>= 1)"},
    {0x1dfa531, "fence wait (!= value)"},
};

struct engine_tls
{
    struct d3d12_fence *fence;
    uint64_t value;
    uint64_t tsc;
    HANDLE event;
    uint64_t last_spin_ns;
    uint64_t wait_ns;

    /* dropdst: 1 = one of the engine's streaming threads, -1 = another thread, 0 = not looked up yet */

    /* frametrace: draws recorded in a row into one render target */
    struct frametrace_rt
    {
        const struct d3d12_resource *resource;
        uint32_t layer, draws;
        uint32_t width, height, depth, format;
        bool dsv;
        char name[48];
    } run;
};

bool vkd3d_engine_spin_patched;
static DWORD engine_tls_index = TLS_OUT_OF_INDEXES;
static pthread_once_t engine_patch_once = PTHREAD_ONCE_INIT;

static FILE *frametrace_file;
static pthread_mutex_t frametrace_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t frametrace_t0, frametrace_last_stats;
static uint64_t frametrace_frames;
static uint32_t frametrace_draws;
static bool frametrace_detail;
static double frametrace_draws_avg;

#define FRAMETRACE_MAX_MARKERS 4096
static struct
{
    uint32_t hash;
    uint32_t count;
    char name[64];
} frametrace_markers[FRAMETRACE_MAX_MARKERS];
static unsigned int frametrace_marker_count;

#define FRAMETRACE_MAX_RTS 1024
static struct frametrace_rt frametrace_rts[FRAMETRACE_MAX_RTS];
static unsigned int frametrace_rt_count;

static struct engine_tls *engine_tls_get(void)
{
    struct engine_tls *tls;

    if (engine_tls_index == TLS_OUT_OF_INDEXES)
        return NULL;
    if (!(tls = TlsGetValue(engine_tls_index)))
    {
        if (!(tls = vkd3d_calloc(1, sizeof(*tls))))
            return NULL;
        TlsSetValue(engine_tls_index, tls);
    }
    return tls;
}

void vkd3d_engine_note_fence_poll(struct d3d12_fence *fence, uint64_t value)
{
    struct engine_tls *tls;

    if ((tls = engine_tls_get()))
    {
        tls->fence = fence;
        tls->value = value;
        tls->tsc = __rdtsc();
    }
}

static void engine_pause(void)
{
    unsigned int i;

    for (i = 0; i < 50; i++)
        __builtin_ia32_pause();
}

/* Replaces the engine's pause helper at the patched sites. Reached through engine_spin_stub,
 * which preserves every register the original helper (pause loop, clobbers rax) left alone. */
static int engine_site_index(uintptr_t ret)
{
    unsigned int i;

    for (i = 0; engine_image && i < ARRAY_SIZE(engine_spin_sites); i++)
        if ((uintptr_t)(engine_image + engine_spin_sites[i].rva + 5) == ret)
            return i;
    return -1;
}

static void engine_site_add(uintptr_t ret, uint64_t ns)
{
    int i = engine_site_index(ret);

    if (i >= 0)
    {
        __atomic_fetch_add(&engine_site_ns[i], ns, __ATOMIC_RELAXED);
        __atomic_fetch_add(&engine_site_count[i], 1, __ATOMIC_RELAXED);
    }
}

void vkd3d_engine_spin_wait(uintptr_t ret)
{
    struct engine_tls *tls = engine_tls_get();
    struct d3d12_fence *fence;
    uint64_t t0, t1;

    if (!tls)
    {
        engine_pause();
        return;
    }

    fence = __rdtsc() - tls->tsc < ENGINE_POLL_FRESH_TSC ? tls->fence : NULL;
    tls->fence = NULL;

    if (!fence || !engine_ctl.spin_block ||
            (!tls->event && !(tls->event = CreateEventW(NULL, FALSE, FALSE, NULL))))
    {
        /* Original behaviour. Successive calls closer than 100us apart are one spin loop. */
        t1 = vkd3d_get_current_time_ns();
        if (tls->last_spin_ns && t1 - tls->last_spin_ns < 100000)
        {
            tls->wait_ns += t1 - tls->last_spin_ns;
            __atomic_fetch_add(&engine_ctl.spin_ns, t1 - tls->last_spin_ns, __ATOMIC_RELAXED);
            engine_site_add(ret, t1 - tls->last_spin_ns);
        }
        tls->last_spin_ns = t1;
        engine_pause();
        return;
    }

    t0 = vkd3d_get_current_time_ns();
    /* Wake on any progress past the value the loop just saw; the loop decides if it's enough. */
    if (FAILED(d3d12_fence_set_event_on_completion(fence, tls->value + 1, tls->event)))
    {
        engine_pause();
        return;
    }
    if (WaitForSingleObject(tls->event, ENGINE_SPIN_TIMEOUT_MS) == WAIT_TIMEOUT)
        __atomic_fetch_add(&engine_ctl.timeouts, 1, __ATOMIC_RELAXED);
    t1 = vkd3d_get_current_time_ns();

    tls->last_spin_ns = 0;
    tls->wait_ns += t1 - t0;
    engine_site_add(ret, t1 - t0);
    __atomic_fetch_add(&engine_ctl.waits, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&engine_ctl.blocked_ns, t1 - t0, __ATOMIC_RELAXED);
}

void engine_spin_stub(void);
__asm__(
    ".text\n"
    ".p2align 4\n"
    "engine_spin_stub:\n"
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
    /* The engine's return address (patched call site + 5): 6 pushes + 0x88 above it. */
    "    mov 0xb8(%rsp), %rcx\n"
    "    call vkd3d_engine_spin_wait\n"
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
    "    ret\n");

/* rel32 calls need their target within 2 GiB, so put a jump to the stub next to the exe. */
static uint8_t *engine_alloc_cave(uint8_t *image, uint32_t image_size)
{
    MEMORY_BASIC_INFORMATION info;
    uint8_t *addr, *cave;
    int dir;

    for (dir = 0; dir < 2; dir++)
    {
        addr = dir ? image - 0x100000 : image + image_size + 0x100000;
        while (addr > image - 0x70000000ull && addr < image + 0x70000000ull)
        {
            if (VirtualQuery(addr, &info, sizeof(info)) && info.State == MEM_FREE &&
                    (cave = VirtualAlloc(addr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)))
                return cave;
            addr = dir ? addr - 0x100000 : addr + 0x100000;
        }
    }
    return NULL;
}

static void engine_patch_spin(uint8_t *image, uint32_t image_size)
{
    uint8_t *cave, *site;
    DWORD old_protect;
    int32_t rel;
    unsigned int i;

    if (engine_tls_index == TLS_OUT_OF_INDEXES)
        return;
    if (!(cave = engine_alloc_cave(image, image_size)))
    {
        ERR("engine patch: no cave within 2 GiB of the exe.\n");
        return;
    }

    /* jmp qword ptr [rip+0]; dq engine_spin_stub */
    cave[0] = 0xff;
    cave[1] = 0x25;
    memset(cave + 2, 0, 4);
    *(void **)(cave + 6) = (void *)engine_spin_stub;
    FlushInstructionCache(GetCurrentProcess(), cave, 14);

    /* The hook must see fence polls before the first patched call can run. */
    engine_image = image;
    vkd3d_engine_spin_patched = true;

    for (i = 0; i < ARRAY_SIZE(engine_spin_sites); i++)
    {
        site = image + engine_spin_sites[i].rva;
        memcpy(&rel, site + 1, sizeof(rel));
        if (site[0] != 0xe8 || site + 5 + rel != image + ENGINE_PAUSE_HELPER)
        {
            ERR("engine patch: unexpected bytes at %#x (%s), skipped.\n",
                    engine_spin_sites[i].rva, engine_spin_sites[i].name);
            continue;
        }

        rel = (int32_t)(cave - (site + 5));
        VirtualProtect(site, 5, PAGE_EXECUTE_READWRITE, &old_protect);
        memcpy(site + 1, &rel, sizeof(rel));
        VirtualProtect(site, 5, old_protect, &old_protect);
        FlushInstructionCache(GetCurrentProcess(), site, 5);
        engine_ctl.sites_patched++;
    }

    INFO("engine patch: spin -> blocking fence wait at %d/%u sites.\n",
            (int)engine_ctl.sites_patched, (unsigned int)ARRAY_SIZE(engine_spin_sites));
}

/* Experiment only, not lossless in general: LightGridManager::Init caps its paged "Light Grid
 * Data Buffer" at min(requested, 0x1fff) pages of 64 KiB (512 MiB of VRAM). VKD3D_BENCH_LIGHTGRID_PAGES
 * lowers that cap, to measure what the freed VRAM is worth. A scene that needs more pages than
 * the cap would lose lights. */
static void engine_patch_lightgrid_cap(uint8_t *image)
{
    static const uint8_t expect[] = {0x41, 0xbe, 0xff, 0x1f, 0x00, 0x00}; /* mov r14d, 0x1fff */
    uint8_t *site = image + 0x1c2d856;
    DWORD old_protect;
    uint32_t pages;
    char env[16];

    if (!vkd3d_get_env_var("VKD3D_BENCH_LIGHTGRID_PAGES", env, sizeof(env)) || !(pages = atoi(env)) ||
            pages >= 0x1fff)
        return;
    if (memcmp(site, expect, sizeof(expect)))
    {
        ERR("engine patch: light grid cap site does not match, not patching.\n");
        return;
    }
    VirtualProtect(site, sizeof(expect), PAGE_EXECUTE_READWRITE, &old_protect);
    memcpy(site + 2, &pages, sizeof(pages));
    VirtualProtect(site, sizeof(expect), old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), site, sizeof(expect));
    INFO("engine patch: light grid capped at %u pages (%u MiB).\n", pages, pages >> 4);
}

/* gdec: the vertex decoder at 0x1c97030 (4 vertices per call: int16 positions + 10:10:10 normals
 * -> 48-byte floats; ~55 call sites, job threads and the render thread) reads its 64-byte source
 * block with four movups straight out of the geometry heap, which is write-combined. WC reads are
 * uncached: four serialized DRAM reads per block, ~844 cycles each, a quarter of the game's
 * sampled load-stall time. MOVNTDQA on WC memory fetches the whole line into a streaming load
 * buffer once and serves the other loads from it. Same bytes loaded into the same registers.
 * MOVNTDQA needs 16-byte alignment; every sampled source was aligned, and an unaligned one takes
 * the original instructions. Flags are dead from the site to the epilogue. */
static volatile uint8_t *engine_gdec_off; /* in the cave page; 1 = take the original loads */

static void engine_patch_gdec(uint8_t *image, uint32_t image_size)
{
    static const uint8_t expect[] =
    {
        0x0f, 0x10, 0x62, 0x20,             /* 0x1c9703b movups xmm4, [rdx + 0x20] */
        0x0f, 0x10, 0x5a, 0x30,             /*           movups xmm3, [rdx + 0x30] */
        0x0f, 0x10, 0x6a, 0x10,             /*           movups xmm5, [rdx + 0x10] */
        0x0f, 0x29, 0x70, 0xe8,             /* 0x1c97047 movaps [rax - 0x18], xmm6 */
        0x0f, 0x29, 0x78, 0xd8,             /*           movaps [rax - 0x28], xmm7 */
        0x44, 0x0f, 0x29, 0x40, 0xc8,       /*           movaps [rax - 0x38], xmm8 */
        0x44, 0x0f, 0x29, 0x48, 0xb8,       /*           movaps [rax - 0x48], xmm9 */
        0x44, 0x0f, 0x29, 0x50, 0xa8,       /*           movaps [rax - 0x58], xmm10 */
        0x44, 0x0f, 0x29, 0x58, 0x98,       /*           movaps [rax - 0x68], xmm11 */
        0x44, 0x0f, 0x29, 0x24, 0x24,       /*           movaps [rsp], xmm12 */
        0x44, 0x0f, 0x10, 0x22,             /* 0x1c97068 movups xmm12, [rdx] */
    };
    static const uint8_t fast_loads[] =
    {
        0x66, 0x0f, 0x38, 0x2a, 0x62, 0x20, /* movntdqa xmm4, [rdx + 0x20] */
        0x66, 0x0f, 0x38, 0x2a, 0x5a, 0x30, /* movntdqa xmm3, [rdx + 0x30] */
        0x66, 0x0f, 0x38, 0x2a, 0x6a, 0x10, /* movntdqa xmm5, [rdx + 0x10] */
    };
    static const uint8_t last_load[] = {0x66, 0x44, 0x0f, 0x38, 0x2a, 0x22}; /* movntdqa xmm12, [rdx] */
    uint8_t *site = image + 0x1c9703b, *back = image + 0x1c9706c, *cave, *p, *jnz1, *jnz2, *slow;
    DWORD old_protect;
    int32_t rel;

    if (memcmp(site, expect, sizeof(expect)))
    {
        ERR("engine patch: vertex decoder site does not match, not patching.\n");
        return;
    }
    if (!(cave = engine_alloc_cave(image, image_size)))
    {
        ERR("engine patch: no cave within 2 GiB of the exe for gdec.\n");
        return;
    }
    engine_gdec_off = cave + 0x800;
    *engine_gdec_off = 0;

    p = cave;
    *p++ = 0xf6; *p++ = 0xc2; *p++ = 0x0f;  /* test dl, 0xf */
    jnz1 = p; p += 2;                       /* jnz slow */
    *p++ = 0x80; *p++ = 0x3d;               /* cmp byte [rip + off_flag], 0 */
    rel = (int32_t)((uint8_t *)engine_gdec_off - (p + 5));
    memcpy(p, &rel, 4); p += 4;
    *p++ = 0x00;
    jnz2 = p; p += 2;                       /* jnz slow */
    memcpy(p, fast_loads, sizeof(fast_loads)); p += sizeof(fast_loads);
    memcpy(p, expect + 12, 33); p += 33;    /* the xmm6-xmm12 saves; xmm12 is saved before it is loaded */
    memcpy(p, last_load, sizeof(last_load)); p += sizeof(last_load);
    *p = 0xe9; rel = (int32_t)(back - (p + 5)); memcpy(p + 1, &rel, 4); p += 5;
    slow = p;                               /* the original instructions */
    memcpy(p, expect, sizeof(expect)); p += sizeof(expect);
    *p = 0xe9; rel = (int32_t)(back - (p + 5)); memcpy(p + 1, &rel, 4); p += 5;
    jnz1[0] = 0x75; jnz1[1] = (uint8_t)(slow - (jnz1 + 2));
    jnz2[0] = 0x75; jnz2[1] = (uint8_t)(slow - (jnz2 + 2));
    FlushInstructionCache(GetCurrentProcess(), cave, p - cave);

    VirtualProtect(site, sizeof(expect), PAGE_EXECUTE_READWRITE, &old_protect);
    memset(site, 0xcc, sizeof(expect));
    site[0] = 0xe9;
    rel = (int32_t)(cave - (site + 5));
    memcpy(site + 1, &rel, 4);
    VirtualProtect(site, sizeof(expect), old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), site, sizeof(expect));
    INFO("engine patch: vertex decoder reads the geometry heap with streaming loads (cave %p).\n", cave);
}

/* dropdst: the engine's streaming pipeline writes into big arenas whose pages have often gone to
 * zram since their last use: LZ4 block decompression (0x20cc4f0, NxStorage Command thread) and
 * memcpy (0x3b59ee0, Asset Load and NxStorage Copy threads). A store into a swapped-out page first
 * swaps the page's dead contents back in (a major fault: zram decompression, and reclaim to make
 * room for it). v6-fin1: 118K of the 210K major faults of the save load were such stores, plus
 * memset (0x3b5a590, rep stosb) with 6.6K.
 * Before a copy, the destination pages it overwrites completely and that are swapped out
 * (/proc/self/pagemap: swapped, not present) get MEM_RESET (Wine: madvise(MADV_DONTNEED)): the
 * swap slot is freed and the store gets a fresh zero page. Every byte of those pages is then
 * written by the copy, so the engine reads exactly what it would have. Present and never-touched
 * pages are left alone, and so are copies whose source overlaps the destination.
 * For LZ4 the overwritten range is the decoded size, which the hook works out from the block
 * first, through the decoder's own checks (engine_lz4_size); the decoder's output never depends on
 * what the destination held before. Live switch: features bit 0x800 turns the resets off. */
static HANDLE engine_pagemap = INVALID_HANDLE_VALUE;
static size_t engine_dropdst_min = 16384;
static struct
{
    uint64_t copies, fills, pinned, noswap, lz4, lz4_swapped, lz4_bad, reads, read_errors, pages, swapped, dropped, reset_errors, ns;
} engine_dropdst_stat;
static uint64_t engine_dropdst_last_log;

#define DROPDST_ADD(field, n) __atomic_fetch_add(&engine_dropdst_stat.field, (n), __ATOMIC_RELAXED)

/* Host-visible memory vkd3d mapped (the driver pins it, so it is never swapped): copies into it skip
 * the pagemap read. An entry left over after an unmap only costs a missed reset, never a wrong one. */
static SRWLOCK engine_pinned_lock = SRWLOCK_INIT;
static struct engine_range { uintptr_t lo, hi; } *engine_pinned;
static size_t engine_pinned_count, engine_pinned_size;

void vkd3d_bench_pinned_add(const void *ptr, uint64_t size)
{
    uintptr_t lo = (uintptr_t)ptr;
    size_t i;

    if (!ptr || !size)
        return;
    AcquireSRWLockExclusive(&engine_pinned_lock);
    if (vkd3d_array_reserve((void **)&engine_pinned, &engine_pinned_size, engine_pinned_count + 1, sizeof(*engine_pinned)))
    {
        for (i = engine_pinned_count; i && engine_pinned[i - 1].lo > lo; i--)
            engine_pinned[i] = engine_pinned[i - 1];
        engine_pinned[i].lo = lo;
        engine_pinned[i].hi = lo + size;
        engine_pinned_count++;
    }
    ReleaseSRWLockExclusive(&engine_pinned_lock);
}

void vkd3d_bench_pinned_remove(const void *ptr)
{
    uintptr_t lo = (uintptr_t)ptr;
    size_t i;

    AcquireSRWLockExclusive(&engine_pinned_lock);
    for (i = 0; i < engine_pinned_count; i++)
    {
        if (engine_pinned[i].lo == lo)
        {
            memmove(&engine_pinned[i], &engine_pinned[i + 1], (engine_pinned_count - i - 1) * sizeof(*engine_pinned));
            engine_pinned_count--;
            break;
        }
    }
    ReleaseSRWLockExclusive(&engine_pinned_lock);
}

static bool engine_pinned_contains(uintptr_t lo, uintptr_t hi)
{
    size_t a = 0, b, m;
    bool ret = false;

    AcquireSRWLockShared(&engine_pinned_lock);
    b = engine_pinned_count;
    while (a < b)
    {
        m = (a + b) / 2;
        if (engine_pinned[m].lo <= lo)
            a = m + 1;
        else
            b = m;
    }
    ret = a && lo < engine_pinned[a - 1].hi && hi <= engine_pinned[a - 1].hi;
    ReleaseSRWLockShared(&engine_pinned_lock);
    return ret;
}

/* Whether the game has anything in swap (VmSwap in /proc/<pid>/status), read again at most every
 * 100 ms. With nothing swapped no page can need a reset, so the pagemap reads are skipped: on a
 * machine with RAM to spare, always. A stale answer only costs a missed reset, never a wrong one. */
static HANDLE engine_status = INVALID_HANDLE_VALUE;
static uint64_t engine_swap_checked_ns;
static bool engine_swap_used = true;

static void __attribute__((noinline)) engine_swap_refresh(void)
{
    char buf[2048], *line;
    OVERLAPPED ov;
    DWORD got;

    memset(&ov, 0, sizeof(ov));
    if (!ReadFile(engine_status, buf, sizeof(buf) - 1, &got, &ov) || !got)
        return;
    buf[got] = 0;
    if ((line = strstr(buf, "\nVmSwap:")))
        __atomic_store_n(&engine_swap_used, strtoull(line + 8, NULL, 10) != 0, __ATOMIC_RELAXED);
}

static bool engine_swap_in_use(void)
{
    uint64_t now, at;

    if (engine_status == INVALID_HANDLE_VALUE)
        return true;
    now = vkd3d_get_current_time_ns();
    at = __atomic_load_n(&engine_swap_checked_ns, __ATOMIC_RELAXED);
    if (now - at >= 100000000ull &&
            __atomic_compare_exchange_n(&engine_swap_checked_ns, &at, now, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
        engine_swap_refresh();
    return __atomic_load_n(&engine_swap_used, __ATOMIC_RELAXED);
}

static void engine_reset_pages(uintptr_t first, uintptr_t end)
{
    if (VirtualAlloc((void *)(first << 12), (end - first) << 12, MEM_RESET, PAGE_READWRITE))
        DROPDST_ADD(dropped, end - first);
    else
        DROPDST_ADD(reset_errors, 1);
}

/* Counts the swapped-out pages in [lo, hi) (page aligned) and, with drop, resets them. */
static unsigned int engine_pagemap_swapped(uintptr_t lo, uintptr_t hi, bool drop)
{
    uintptr_t page, end, n, i, run = 0;
    unsigned int swapped = 0;
    uint64_t pm[128], t0; /* small: memcpy can run on any engine stack */
    OVERLAPPED ov;
    DWORD got;

    if (hi <= lo || engine_pagemap == INVALID_HANDLE_VALUE)
        return 0;
    if (!engine_swap_in_use())
    {
        DROPDST_ADD(noswap, 1);
        return 0;
    }
    if (engine_pinned_contains(lo, hi))
    {
        DROPDST_ADD(pinned, 1);
        return 0;
    }
    t0 = vkd3d_get_current_time_ns();
    for (page = lo >> 12, end = hi >> 12; page < end; page += n)
    {
        n = min(end - page, ARRAY_SIZE(pm));
        memset(&ov, 0, sizeof(ov));
        ov.Offset = (DWORD)(page * sizeof(*pm));
        ov.OffsetHigh = (DWORD)((page * sizeof(*pm)) >> 32);
        DROPDST_ADD(reads, 1);
        if (!ReadFile(engine_pagemap, pm, n * sizeof(*pm), &got, &ov) || got != n * sizeof(*pm))
        {
            DROPDST_ADD(read_errors, 1);
            break;
        }
        for (i = 0; i < n; i++)
        {
            /* bit 63 present, bit 62 swapped */
            if ((pm[i] >> 62) == 1)
            {
                swapped++;
                if (!run)
                    run = page + i;
            }
            else if (run)
            {
                if (drop)
                    engine_reset_pages(run, page + i);
                run = 0;
            }
        }
    }
    if (run && drop)
        engine_reset_pages(run, page);
    DROPDST_ADD(pages, (hi - lo) >> 12);
    DROPDST_ADD(swapped, swapped);
    DROPDST_ADD(ns, vkd3d_get_current_time_ns() - t0);
    return swapped;
}

static void engine_dropdst_memcpy(uint8_t *dst, const uint8_t *src, size_t size)
{
    uintptr_t d = (uintptr_t)dst, s = (uintptr_t)src;

    if ((engine_ctl.features & VKD3D_BENCH_FEATURE_DROPDST_OFF) || (s < d + size && d < s + size))
        return;
    DROPDST_ADD(copies, 1);
    engine_pagemap_swapped((d + 0xfff) & ~(uintptr_t)0xfff, (d + size) & ~(uintptr_t)0xfff, true);
}

static void engine_dropdst_memset(uint8_t *dst, int value, size_t size)
{
    uintptr_t d = (uintptr_t)dst;

    if (engine_ctl.features & VKD3D_BENCH_FEATURE_DROPDST_OFF)
        return;
    DROPDST_ADD(fills, 1);
    engine_pagemap_swapped((d + 0xfff) & ~(uintptr_t)0xfff, (d + size) & ~(uintptr_t)0xfff, true);
}

/* The decoded size of an LZ4 block as the engine's decoder (0x20cc4f0: LZ4_decompress_safe without
 * dictionary, cap > 0) produces it, through the same checks in the same order; -1 where the
 * decoder fails. Reads the block only. */
static int64_t engine_lz4_size(const uint8_t *src, int src_size, uint8_t *dst, int dst_cap)
{
    uintptr_t ip = (uintptr_t)src, iend = ip + src_size, op = (uintptr_t)dst, oend = op + dst_cap;
    uintptr_t length, cpy, offset;
    unsigned int token, s;

    for (;;)
    {
        token = *(const uint8_t *)ip++;
        if ((length = token >> 4) == 15)
        {
            do
            {
                s = *(const uint8_t *)ip++;
                length += s;
                if (ip >= iend - 15)
                    break;
            } while (s == 255);
            if (op + length < op || ip + length < ip)
                return -1;
        }
        cpy = op + length;
        if (cpy > oend - 12 || ip + length > iend - 8)
        {
            if (ip + length == iend && cpy <= oend)
                return cpy - (uintptr_t)dst;
            return -1;
        }
        ip += length;
        op = cpy;
        offset = *(const uint8_t *)ip | (*(const uint8_t *)(ip + 1) << 8);
        ip += 2;
        if (op - offset < (uintptr_t)dst)
            return -1;
        if ((length = token & 15) == 15)
        {
            do
            {
                if (ip > iend - 5)
                    return -1;
                s = *(const uint8_t *)ip++;
                length += s;
            } while (s == 255);
            if (op + length < op)
                return -1;
        }
        cpy = op + length + 4;
        if (cpy > oend - 5)
            return -1;
        op = cpy;
    }
}

static void engine_dropdst_lz4(const uint8_t *src, uint8_t *dst, int src_size, int dst_cap)
{
    uintptr_t d = (uintptr_t)dst, s = (uintptr_t)src, lo = (d + 0xfff) & ~(uintptr_t)0xfff;
    int64_t size;

    if ((engine_ctl.features & VKD3D_BENCH_FEATURE_DROPDST_OFF) || src_size <= 0 ||
            (s < d + dst_cap && d < s + src_size))
        return;
    DROPDST_ADD(lz4, 1);
    if (!engine_pagemap_swapped(lo, (d + dst_cap) & ~(uintptr_t)0xfff, false))
        return;
    DROPDST_ADD(lz4_swapped, 1);
    if ((size = engine_lz4_size(src, src_size, dst, dst_cap)) < 0)
    {
        DROPDST_ADD(lz4_bad, 1);
        return;
    }
    engine_pagemap_swapped(lo, (d + size) & ~(uintptr_t)0xfff, true);
}

/* Calls *rax with the function's own arguments (rcx, rdx, r8, r9), keeping every register the
 * patched function can still read: the argument registers, r10, r11 and xmm0-5. Any stack
 * alignment. */
void engine_hook_stub(void);
__asm__(
    ".text\n"
    ".p2align 4\n"
    "engine_hook_stub:\n"
    "    push %rbp\n"
    "    mov %rsp, %rbp\n"
    "    push %rcx\n"
    "    push %rdx\n"
    "    push %r8\n"
    "    push %r9\n"
    "    push %r10\n"
    "    push %r11\n"
    "    and $-16, %rsp\n"
    "    sub $0x80, %rsp\n"
    "    movdqu %xmm0, 0x20(%rsp)\n"
    "    movdqu %xmm1, 0x30(%rsp)\n"
    "    movdqu %xmm2, 0x40(%rsp)\n"
    "    movdqu %xmm3, 0x50(%rsp)\n"
    "    movdqu %xmm4, 0x60(%rsp)\n"
    "    movdqu %xmm5, 0x70(%rsp)\n"
    "    call *%rax\n"
    "    movdqu 0x20(%rsp), %xmm0\n"
    "    movdqu 0x30(%rsp), %xmm1\n"
    "    movdqu 0x40(%rsp), %xmm2\n"
    "    movdqu 0x50(%rsp), %xmm3\n"
    "    movdqu 0x60(%rsp), %xmm4\n"
    "    movdqu 0x70(%rsp), %xmm5\n"
    "    lea -0x30(%rbp), %rsp\n"
    "    pop %r11\n"
    "    pop %r10\n"
    "    pop %r9\n"
    "    pop %r8\n"
    "    pop %rdx\n"
    "    pop %rcx\n"
    "    pop %rbp\n"
    "    ret\n");

/* Diverts a function entry to `cave` while the engine runs: a jmp rel32 goes into the never-executed
 * int3 padding at `pad`, then the entry's first two bytes become a short jmp to it in one aligned
 * 2-byte store. The entry's first instruction is longer than two bytes, so no thread can be
 * stopped between the two bytes; the cave runs that instruction itself. */
static bool engine_divert_entry(uint8_t *entry, uint8_t *pad, uint8_t *cave)
{
    DWORD old_protect;
    int32_t rel;
    unsigned int i;

    for (i = 0; i < 5; i++)
        if (pad[i] != 0xcc)
            return false;
    if (((uintptr_t)entry & 1) || pad + 5 > entry || entry + 2 - pad > 128)
        return false;
    VirtualProtect(pad, entry + 2 - pad, PAGE_EXECUTE_READWRITE, &old_protect);
    pad[0] = 0xe9;
    rel = (int32_t)(cave - (pad + 5));
    memcpy(pad + 1, &rel, 4);
    FlushInstructionCache(GetCurrentProcess(), pad, 5);
    __atomic_store_n((uint16_t *)entry, (uint16_t)(0xeb | ((uint8_t)(pad - (entry + 2)) << 8)), __ATOMIC_SEQ_CST);
    VirtualProtect(pad, entry + 2 - pad, old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), entry, 2);
    return true;
}

static uint8_t *engine_cave_call_hook(uint8_t *p, void *hook)
{
    int32_t rel;

    *p++ = 0x48; *p++ = 0xb8;               /* movabs rax, hook */
    memcpy(p, &hook, 8); p += 8;
    *p++ = 0xff; *p++ = 0x15;               /* call [rip + 2] */
    rel = 2;
    memcpy(p, &rel, 4); p += 4;
    *p++ = 0xeb; *p++ = 0x08;               /* jmp over the pointer */
    *(void **)p = (void *)engine_hook_stub; p += 8;
    return p;
}

static bool engine_read_file(const WCHAR *path, void *buf, DWORD size, uint64_t offset, DWORD *got)
{
    OVERLAPPED ov = {0};
    HANDLE h;
    BOOL ret;

    if ((h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL))
            == INVALID_HANDLE_VALUE)
        return false;
    ov.Offset = (DWORD)offset;
    ov.OffsetHigh = (DWORD)(offset >> 32);
    ret = ReadFile(h, buf, size, got, &ov);
    CloseHandle(h);
    return ret;
}

/* Wine opens files in the wineserver, so /proc/self there is the wineserver's. This process's
 * pagemap is the /proc/<pid> whose comm is this exe's name and whose pagemap shows a page touched
 * here as present and the reserved page after it as absent. */
static HANDLE engine_open_pagemap(unsigned int *pid_out)
{
    WCHAR exe[MAX_PATH], path[64], *name;
    HANDLE find, h = INVALID_HANDLE_VALUE;
    char comm[32], want[16];
    WIN32_FIND_DATAW fd;
    uint64_t pm[2];
    uint8_t *probe;
    unsigned int i;
    DWORD got;

    if (!GetModuleFileNameW(NULL, exe, ARRAY_SIZE(exe)))
        return INVALID_HANDLE_VALUE;
    name = wcsrchr(exe, '\\') ? wcsrchr(exe, '\\') + 1 : exe;
    for (i = 0; i < 15 && name[i]; i++)
        want[i] = (char)name[i];
    want[i] = 0;
    if (!(probe = VirtualAlloc(NULL, 0x10000, MEM_RESERVE, PAGE_READWRITE)))
        return INVALID_HANDLE_VALUE;
    VirtualAlloc(probe, 0x1000, MEM_COMMIT, PAGE_READWRITE);
    *(volatile uint8_t *)probe = 1;
    if ((find = FindFirstFileW(L"\\\\?\\unix\\proc\\*", &fd)) != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (!iswdigit(fd.cFileName[0]) || wcslen(fd.cFileName) > 10)
                continue;
            swprintf(path, ARRAY_SIZE(path), L"\\\\?\\unix\\proc\\%ls\\comm", fd.cFileName);
            memset(comm, 0, sizeof(comm));
            if (!engine_read_file(path, comm, sizeof(comm) - 1, 0, &got))
                continue;
            comm[strcspn(comm, "\n")] = 0;
            if (strcmp(comm, want))
                continue;
            swprintf(path, ARRAY_SIZE(path), L"\\\\?\\unix\\proc\\%ls\\pagemap", fd.cFileName);
            if (!engine_read_file(path, pm, sizeof(pm), ((uintptr_t)probe >> 12) * sizeof(*pm), &got) ||
                    got != sizeof(pm) || (pm[0] >> 63) != 1 || (pm[1] >> 62) != 0)
                continue;
            h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
            *pid_out = wcstoul(fd.cFileName, NULL, 10);
            break;
        } while (FindNextFileW(find, &fd));
        FindClose(find);
    }
    VirtualFree(probe, 0, MEM_RELEASE);
    return h;
}

static void engine_patch_dropdst(uint8_t *image, uint32_t image_size)
{
    static const uint8_t memcpy_expect[] = {0x48, 0x8b, 0xc1, 0x4c, 0x8d, 0x15}; /* mov rax, rcx; lea r10, [rip + ...] */
    static const uint8_t lz4_expect[] = {0x4c, 0x8b, 0xdc, 0x49, 0x89, 0x5b, 0x18}; /* mov r11, rsp; mov [r11 + 0x18], rbx */
    static const uint8_t memset_expect[] = {0x4c, 0x8b, 0xd9, 0x0f, 0xb6, 0xd2}; /* mov r11, rcx; movzx edx, dl */
    uint8_t *mc = image + 0x3b59ee0, *lz = image + 0x20cc4f0, *ms = image + 0x3b5a590, *cave, *p, *jb, *lc, *jl, *sc;
    unsigned int pid = 0;
    WCHAR path[64];
    uint32_t min;
    int32_t rel;
    char env[32];

    if (memcmp(mc, memcpy_expect, sizeof(memcpy_expect)) || memcmp(lz, lz4_expect, sizeof(lz4_expect)) ||
            memcmp(ms, memset_expect, sizeof(memset_expect)))
    {
        ERR("engine patch: memcpy/memset/LZ4 entries do not match, not patching.\n");
        return;
    }
    if ((engine_pagemap = engine_open_pagemap(&pid)) == INVALID_HANDLE_VALUE)
    {
        ERR("engine patch: did not find this process's /proc/<pid>/pagemap, not patching dropdst.\n");
        return;
    }
    swprintf(path, ARRAY_SIZE(path), L"\\\\?\\unix\\proc\\%u\\status", pid);
    if ((engine_status = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL))
            == INVALID_HANDLE_VALUE)
        ERR("engine patch: cannot read /proc/%u/status; dropdst checks the pagemap on every copy.\n", pid);
    if (vkd3d_get_env_var("VKD3D_BENCH_DROPDST_MIN", env, sizeof(env)) && atoi(env) >= 8192)
        engine_dropdst_min = atoi(env);
    min = engine_dropdst_min;
    if (!(cave = engine_alloc_cave(image, image_size)))
    {
        ERR("engine patch: no cave within 2 GiB of the exe for dropdst.\n");
        return;
    }

    /* memcpy(rcx dst, rdx src, r8 size): sizes >= min call the hook first. */
    p = cave;
    *p++ = 0x48; *p++ = 0x8b; *p++ = 0xc1;  /* mov rax, rcx */
    *p++ = 0x49; *p++ = 0x81; *p++ = 0xf8;  /* cmp r8, min */
    memcpy(p, &min, 4); p += 4;
    jb = p; p += 6;                         /* jb back */
    p = engine_cave_call_hook(p, (void *)engine_dropdst_memcpy);
    *p++ = 0x48; *p++ = 0x8b; *p++ = 0xc1;  /* mov rax, rcx */
    jb[0] = 0x0f; jb[1] = 0x82;
    rel = (int32_t)((mc + 3) - (jb + 6));
    memcpy(jb + 2, &rel, 4);
    *p = 0xe9; rel = (int32_t)((mc + 3) - (p + 5)); memcpy(p + 1, &rel, 4); p += 5;

    /* LZ4(rcx src, rdx dst, r8d src size, r9d dst capacity): capacities >= min call the hook first. */
    lc = p = (uint8_t *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
    *p++ = 0x41; *p++ = 0x81; *p++ = 0xf9;  /* cmp r9d, min */
    memcpy(p, &min, 4); p += 4;
    jl = p; p += 6;                         /* jl tail */
    p = engine_cave_call_hook(p, (void *)engine_dropdst_lz4);
    jl[0] = 0x0f; jl[1] = 0x8c;
    rel = (int32_t)(p - (jl + 6));
    memcpy(jl + 2, &rel, 4);
    *p++ = 0x4c; *p++ = 0x8b; *p++ = 0xdc;  /* tail: mov r11, rsp */
    *p = 0xe9; rel = (int32_t)((lz + 3) - (p + 5)); memcpy(p + 1, &rel, 4); p += 5;

    /* memset(rcx dst, dl value, r8 size): sizes >= min call the hook first. */
    sc = p = (uint8_t *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
    *p++ = 0x4c; *p++ = 0x8b; *p++ = 0xd9;  /* mov r11, rcx */
    *p++ = 0x49; *p++ = 0x81; *p++ = 0xf8;  /* cmp r8, min */
    memcpy(p, &min, 4); p += 4;
    jb = p; p += 6;                         /* jb back */
    p = engine_cave_call_hook(p, (void *)engine_dropdst_memset);
    jb[0] = 0x0f; jb[1] = 0x82;
    rel = (int32_t)((ms + 3) - (jb + 6));
    memcpy(jb + 2, &rel, 4);
    *p = 0xe9; rel = (int32_t)((ms + 3) - (p + 5)); memcpy(p + 1, &rel, 4); p += 5;
    FlushInstructionCache(GetCurrentProcess(), cave, p - cave);

    if (!engine_divert_entry(mc, image + 0x3b59ed0, cave) || !engine_divert_entry(lz, image + 0x20cc4e8, lc) ||
            !engine_divert_entry(ms, image + 0x3b5a580, sc))
    {
        ERR("engine patch: dropdst padding does not match.\n");
        return;
    }
    INFO("engine patch: dropdst, swapped destination pages of memcpy, memset and LZ4 copies >= %u bytes are reset first "
            "(cave %p, pagemap of unix pid %u).\n", min, cave, pid);
}

static void engine_dropdst_log(void)
{
    uint64_t now = vkd3d_get_current_time_ns();

    if (engine_pagemap == INVALID_HANDLE_VALUE || now - engine_dropdst_last_log < 10000000000ull)
        return;
    engine_dropdst_last_log = now;
    INFO("dropdst: %"PRIu64" copies, %"PRIu64" fills, %"PRIu64" into pinned GPU memory, %"PRIu64" skipped with nothing swapped, %"PRIu64" LZ4 blocks (%"PRIu64" onto swapped pages, %"PRIu64" undecodable); "
            "%"PRIu64" pagemap reads (%"PRIu64" failed), %"PRIu64" pages checked, %"PRIu64" swapped, %"PRIu64" reset "
            "(%"PRIu64" reset errors), %.1f ms in the hook.\n",
            engine_dropdst_stat.copies, engine_dropdst_stat.fills, engine_dropdst_stat.pinned, engine_dropdst_stat.noswap,
            engine_dropdst_stat.lz4, engine_dropdst_stat.lz4_swapped,
            engine_dropdst_stat.lz4_bad, engine_dropdst_stat.reads, engine_dropdst_stat.read_errors,
            engine_dropdst_stat.pages, engine_dropdst_stat.swapped, engine_dropdst_stat.dropped,
            engine_dropdst_stat.reset_errors, engine_dropdst_stat.ns / 1e6);
}

/* Retarget the rel32 call at `site_rva` (which must call `target_rva`) to `wrapper`, through an
 * absolute jump in a cave next to the exe. */
static bool engine_redirect_call(uint8_t *image, uint32_t image_size, uint32_t site_rva, uint32_t target_rva,
        void *wrapper)
{
    uint8_t *site = image + site_rva, *cave;
    DWORD old_protect;
    int32_t rel;

    memcpy(&rel, site + 1, 4);
    if (site[0] != 0xe8 || site + 5 + rel != image + target_rva)
    {
        ERR("engine patch: call site %#x does not call %#x, not patching.\n", site_rva, target_rva);
        return false;
    }
    if (!(cave = engine_alloc_cave(image, image_size)))
        return false;
    cave[0] = 0xff;
    cave[1] = 0x25;
    memset(cave + 2, 0, 4);
    *(void **)(cave + 6) = wrapper;
    FlushInstructionCache(GetCurrentProcess(), cave, 14);
    rel = (int32_t)(cave - (site + 5));
    VirtualProtect(site, 5, PAGE_EXECUTE_READWRITE, &old_protect);
    memcpy(site + 1, &rel, 4);
    VirtualProtect(site, 5, old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), site, 5);
    return true;
}

/* Set once every hook a feature relies on is in place (engine_patch_init runs at device creation,
 * before the game creates a heap): without them the index cache and the lazy skinning buffer stay
 * off, so an unknown exe build or a site that did not match leaves those heaps as they were. */
static bool engine_geometry_hooked, engine_skin_hooked;

bool vkd3d_engine_geometry_hooked(void)
{
    return engine_geometry_hooked;
}

bool vkd3d_engine_skin_hooked(void)
{
    return engine_skin_hooked;
}

/* Skinning buffer (skinbuf.c): every call of the engine's bump allocator (0x2162ef0, and its aligned
 * variant 0x2162f20) goes to skinbuf.c's copies, which do the same atomic operations and back the
 * pages of a skinning allocation before returning it. These are all the calls Ghidra finds (no
 * jumps, no data references); 0x2162ef0's four are the skinning manager's (and three other
 * allocators'), 0x2162f20's four are other allocators'. */
static void engine_patch_skin(uint8_t *image, uint32_t image_size)
{
    static const uint32_t alloc_sites[] = {0x1b19f47, 0x1c9597b, 0x1bb8771, 0x1c80f3b};
    static const uint32_t aligned_sites[] = {0x1d36b68, 0x1d36bc8, 0x1d33ed7, 0x1d33eec};
    unsigned int i, done = 0;

    for (i = 0; i < ARRAY_SIZE(alloc_sites); i++)
        done += engine_redirect_call(image, image_size, alloc_sites[i], 0x2162ef0, (void *)vkd3d_skinbuf_alloc_stub);
    for (i = 0; i < ARRAY_SIZE(aligned_sites); i++)
        done += engine_redirect_call(image, image_size, aligned_sites[i], 0x2162f20, (void *)vkd3d_skinbuf_alloc_aligned_stub);
    if (done != ARRAY_SIZE(alloc_sites) + ARRAY_SIZE(aligned_sites))
        ERR("engine patch: only %u of %u bump allocator calls redirected; skinning allocations may go unbacked.\n",
                done, (unsigned int)(ARRAY_SIZE(alloc_sites) + ARRAY_SIZE(aligned_sites)));
    else
    {
        engine_skin_hooked = true;
        INFO("engine patch: %u bump allocator calls go through skinbuf.c.\n", done);
    }
}

/* Index cache invalidation (idxcache.c). The engine puts geometry into the ManagedBuffer heap
 * only through D3DBufferManager: the AllocateBuffer lambda (0x1d6d4a0, called once from
 * AllocateBuffer at 0x1d6d7e7) takes a range and fills it, now or later, through 0x1d6e810
 * (mgr, heap, mapped, offset, data, size), which for this mapped heap stores into
 * [mgr + 0x5b8] + offset. Its three callers: the lambda (0x1d6d5e8), the deferred list flush
 * 0x1d6dba0 (0x1d6dbfd) and render command 0xa1 (0x1d912b0, at 0x1d9130f). Every written range is
 * reported after the write, and every new allocation too. */
typedef void *(*engine_geo_write_fn)(void *mgr, void *heap, void *mapped, uint32_t offset, void *data, uint64_t size);
typedef uint8_t *(*engine_geo_alloc_fn)(void *ctx);
static engine_geo_write_fn engine_geo_write_orig;
static engine_geo_alloc_fn engine_geo_alloc_orig;

static void *engine_geo_write(void *mgr, void *heap, void *mapped, uint32_t offset, void *data, uint64_t size)
{
    void *ret = engine_geo_write_orig(mgr, heap, mapped, offset, data, size);

    vkd3d_index_cache_invalidate_cpu(*(uint8_t **)((uint8_t *)mgr + 0x5b8) + offset, (uint32_t)size);
    vkd3d_geomirror_note_write(*(uint8_t **)((uint8_t *)mgr + 0x5b8) + offset, size);
    return ret;
}

static uint8_t *engine_geo_alloc(void *ctx)
{
    uint8_t *record = engine_geo_alloc_orig(ctx);

    /* {heap, mapped base, ..., +0x14 offset, +0x1c size} */
    if (record)
    {
        vkd3d_index_cache_invalidate_cpu(*(uint8_t **)(record + 8) + *(int32_t *)(record + 0x14),
                *(uint32_t *)(record + 0x1c));
        vkd3d_geomirror_note_write(*(uint8_t **)(record + 8) + *(int32_t *)(record + 0x14),
                *(uint32_t *)(record + 0x1c));
    }
    return record;
}

static void engine_patch_geometry_writes(uint8_t *image, uint32_t image_size)
{
    static const uint32_t write_sites[] = {0x1d6d5e8, 0x1d6dbfd, 0x1d9130f};
    unsigned int i, n = 0;

    engine_geo_write_orig = (engine_geo_write_fn)(image + 0x1d6e810);
    engine_geo_alloc_orig = (engine_geo_alloc_fn)(image + 0x1d6d4a0);
    for (i = 0; i < ARRAY_SIZE(write_sites); i++)
        n += engine_redirect_call(image, image_size, write_sites[i], 0x1d6e810, (void *)engine_geo_write);
    n += engine_redirect_call(image, image_size, 0x1d6d7e7, 0x1d6d4a0, (void *)engine_geo_alloc);
    if (n != ARRAY_SIZE(write_sites) + 1)
    {
        /* A writer we don't see could leave stale copies: no cache then. */
        vkd3d_index_cache_disable("not every geometry writer is hooked");
        return;
    }
    engine_geometry_hooked = true;
    vkd3d_geomirror_hooks_installed();
    INFO("engine patch: geometry writes reported to the index cache and the geometry mirror.\n");
}

/* qdefer: engine layout (MilesMorales.exe 1.1116.0.0). A batch entry is 16 bytes with the query
 * object at +8; the object holds its result block at +0x30 (flags u16 at +0x1e: 1 = has a result,
 * 2 = issued, 4 = finalized; pixel count i32 at +0x14) and the engine's query handle at +0x38.
 * The engine's command context is the first pointer of the PushBuffer at 0x83de920; its vtable
 * slot 0xe8 is GetData(handle, out, size, flags) returning 0 = done, 1 = not yet. */
#define ENGINE_QUERY_FINALIZE_BLOCKING 0x1d98700u
#define ENGINE_PUSHBUFFER 0x83de920u
#define ENGINE_PUSHBUFFER_KICK 0x1df9850u
#define ENGINE_QDEFER_MAX 4096

typedef void (*engine_kick_fn)(void *pushbuffer, uint32_t a, uint32_t b);
typedef int (*engine_getdata_fn)(void *context, void *handle, int64_t *out, uint32_t size, uint32_t flags);

static void *engine_qdefer_list[ENGINE_QDEFER_MAX];
static unsigned int engine_qdefer_count;
static DWORD engine_qdefer_thread;
static bool engine_qdefer_active;
static uint64_t engine_qdefer_stat_batches, engine_qdefer_stat_ready, engine_qdefer_stat_deferred;
static uint64_t engine_qdefer_stat_reissued, engine_qdefer_stat_wait_ns, engine_qdefer_stat_foreign;

static int engine_query_getdata(void *query, int64_t *out)
{
    void *context = *(void **)(engine_image + ENGINE_PUSHBUFFER);
    engine_getdata_fn getdata = (*(engine_getdata_fn **)context)[0xe8 / sizeof(void *)];

    *out = 0;
    return getdata(context, *(void **)((uint8_t *)query + 0x38), out, 8, 1);
}

static void engine_query_store(void *query, int64_t value)
{
    uint8_t *result = *(uint8_t **)((uint8_t *)query + 0x30);

    /* Same as the engine: clamp to INT32_MAX, mark "has a result" and "finalized". */
    *(uint16_t *)(result + 0x1e) |= 5;
    *(int32_t *)(result + 0x14) = value < 0x7fffffff ? (int32_t)value : 0x7fffffff;
}

static bool engine_query_pending(void *query)
{
    uint8_t *result = *(uint8_t **)((uint8_t *)query + 0x30);

    return !(*(uint16_t *)(result + 0x1e) & 4) && *(void **)((uint8_t *)query + 0x38);
}

static void engine_qdefer_flush(void);
static void engine_qasync_hook_table(void);
static bool engine_qasync;

/* Replaces the FinalizeQueriesBlocking handler: (entries, count) like the engine's. */
static void engine_qdefer_finalize(uint8_t *entries, uint32_t count)
{
    engine_kick_fn kick = (engine_kick_fn)(engine_image + ENGINE_PUSHBUFFER_KICK);
    unsigned int i, j;
    int64_t value;
    void *query;
    int hr;

    kick(engine_image + ENGINE_PUSHBUFFER, 1, 1);
    if (!engine_qdefer_thread)
        engine_qdefer_thread = GetCurrentThreadId();
    if (engine_qasync)
        engine_qasync_hook_table();
    engine_qdefer_stat_batches++;

    for (i = 0; i < count; i++)
    {
        query = *(void **)(entries + 16 * i + 8);
        if (!engine_query_pending(query))
            continue;
        hr = engine_query_getdata(query, &value);
        if (hr == 0)
        {
            engine_query_store(query, value);
            engine_qdefer_stat_ready++;
            continue;
        }
        if (hr != 1)
            continue; /* the engine gives up on errors too */
        if (engine_qasync && !(engine_ctl.features & VKD3D_BENCH_FEATURE_QASYNC))
        {
            /* qasync switched off live: the engine's own blocking finalize. */
            while ((hr = engine_query_getdata(query, &value)) == 1)
                engine_pause();
            if (hr == 0)
                engine_query_store(query, value);
            continue;
        }

        for (j = 0; j < engine_qdefer_count; j++)
            if (engine_qdefer_list[j] == query)
                break;
        if (j < engine_qdefer_count)
        {
            /* Re-issued since it was deferred: its earlier answer is gone. Count it. */
            engine_qdefer_stat_reissued++;
            continue;
        }
        if (engine_qdefer_count == ENGINE_QDEFER_MAX)
            engine_qdefer_flush(); /* never in practice: ~40 queries a frame */
        engine_qdefer_list[engine_qdefer_count++] = query;
        engine_qdefer_stat_deferred++;
    }
}

/* End of the render thread's frame: the blocking finalize the engine would have done mid-frame. */
/* Finish deferred queries. block = wait for the ones the GPU hasn't answered yet (the engine's
 * behaviour); otherwise take what is ready and keep the rest for later. */
static void engine_qdefer_poll(bool block)
{
    unsigned int i, kept = 0;
    uint64_t t0 = 0;
    int64_t value;
    void *query;
    int hr;

    if (!engine_qdefer_count)
        return;
    if (GetCurrentThreadId() != engine_qdefer_thread)
        engine_qdefer_stat_foreign++; /* shouldn't happen: all hooks run on the render thread */
    for (i = 0; i < engine_qdefer_count; i++)
    {
        query = engine_qdefer_list[i];
        if (!engine_query_pending(query))
            continue;
        while ((hr = engine_query_getdata(query, &value)) == 1 && block)
        {
            if (!t0)
                t0 = vkd3d_get_current_time_ns();
            engine_pause();
        }
        if (hr == 0)
            engine_query_store(query, value);
        else if (hr == 1)
            engine_qdefer_list[kept++] = query;
    }
    if (t0)
        engine_qdefer_stat_wait_ns += vkd3d_get_current_time_ns() - t0;
    engine_qdefer_count = kept;
}

static void engine_qdefer_flush(void)
{
    engine_qdefer_poll(true);
}

/* qasync: ProcessQueries (render command 0x74, handler 0x1d98f00) re-issues queries. Before it
 * does, every deferred answer is collected (blocking), so none is lost; by then the GPU is a frame
 * further on and the wait is normally zero. Hooked through the engine's command table at 0x83ca9a8,
 * which the game thread copies to the active table (0x83cd1b8) for every frame. */
#define ENGINE_COMMAND_TABLE 0x83ca9a8u
#define ENGINE_CMD_PROCESS_QUERIES 0x74u
#define ENGINE_PROCESS_QUERIES 0x1d98f00u

typedef void (*engine_command_fn)(void *entries, uint32_t count);
static uint64_t engine_qdefer_stat_process;

static void engine_qasync_process_queries(void *entries, uint32_t count)
{
    engine_qdefer_stat_process++;
    engine_qdefer_poll(true);
    ((engine_command_fn)(engine_image + ENGINE_PROCESS_QUERIES))(entries, count);
}

static void engine_qasync_hook_table(void)
{
    void **slot = (void **)(engine_image + ENGINE_COMMAND_TABLE) + ENGINE_CMD_PROCESS_QUERIES;
    static bool done;

    if (done)
        return;
    done = true;
    if (*slot != engine_image + ENGINE_PROCESS_QUERIES)
    {
        ERR("engine patch: command table slot %#x holds %p, not ProcessQueries; qasync stays blocking.\n",
                ENGINE_CMD_PROCESS_QUERIES, *slot);
        engine_qasync = false;
        return;
    }
    *slot = (void *)engine_qasync_process_queries;
    INFO("engine patch: qasync hooked ProcessQueries in the command table.\n");
}

/* The render loop calls PushBuffer::WaitForFrame (0x1d7c5d0) at 0x1d9feb3 at the end of each
 * frame, on the render thread, before it hands the frame back to the game thread. qdefer redirects
 * that call here: finish the deferred queries first, then the engine's own wait. */
#define ENGINE_WAIT_FOR_FRAME 0x1d7c5d0u
#define ENGINE_WAIT_FOR_FRAME_CALL 0x1d9feb3u

typedef void (*engine_wait_for_frame_fn)(void *pushbuffer);

static void engine_qdefer_wait_for_frame(void *pushbuffer)
{
    engine_qdefer_poll(!(engine_qasync && (engine_ctl.features & VKD3D_BENCH_FEATURE_QASYNC)));
    ((engine_wait_for_frame_fn)(engine_image + ENGINE_WAIT_FOR_FRAME))(pushbuffer);
}

static void engine_patch_qdefer(uint8_t *image, uint32_t image_size)
{
    static const uint8_t expect[] = {0x40, 0x53, 0x56, 0x48, 0x83, 0xec, 0x38, 0x41, 0xb0, 0x01};
    uint8_t *site = image + ENGINE_QUERY_FINALIZE_BLOCKING;
    uint8_t jump[14] = {0xff, 0x25, 0, 0, 0, 0};
    void *target = (void *)engine_qdefer_finalize;
    uint8_t *call, *cave;
    DWORD old_protect;
    int32_t rel;

    if (memcmp(site, expect, sizeof(expect)))
    {
        ERR("engine patch: FinalizeQueriesBlocking does not match, not patching.\n");
        return;
    }
    call = image + ENGINE_WAIT_FOR_FRAME_CALL;
    memcpy(&rel, call + 1, sizeof(rel));
    if (call[0] != 0xe8 || call + 5 + rel != image + ENGINE_WAIT_FOR_FRAME)
    {
        ERR("engine patch: WaitForFrame call does not match, not patching qdefer.\n");
        return;
    }
    if (!(cave = engine_alloc_cave(image, image_size)))
    {
        ERR("engine patch: no cave within 2 GiB of the exe for qdefer.\n");
        return;
    }
    cave[0] = 0xff;
    cave[1] = 0x25;
    memset(cave + 2, 0, 4);
    *(void **)(cave + 6) = (void *)engine_qdefer_wait_for_frame;
    FlushInstructionCache(GetCurrentProcess(), cave, 14);

    engine_image = image;
    rel = (int32_t)(cave - (call + 5));
    VirtualProtect(call, 5, PAGE_EXECUTE_READWRITE, &old_protect);
    memcpy(call + 1, &rel, sizeof(rel));
    VirtualProtect(call, 5, old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), call, 5);

    memcpy(jump + 6, &target, sizeof(target));
    VirtualProtect(site, sizeof(jump), PAGE_EXECUTE_READWRITE, &old_protect);
    memcpy(site, jump, sizeof(jump));
    VirtualProtect(site, sizeof(jump), old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), site, sizeof(jump));
    engine_qdefer_active = true;
    INFO("engine patch: FinalizeQueriesBlocking defers unfinished queries to the end of the render frame.\n");
}

void vkd3d_engine_present(void)
{
    if (engine_gdec_off)
        *engine_gdec_off = !!(engine_ctl.features & VKD3D_BENCH_FEATURE_GDEC_OFF);
    engine_dropdst_log();
    vkd3d_skinbuf_log();
}

static void engine_patch_init(void)
{
    char patches[64], path[VKD3D_PATH_MAX];
    IMAGE_NT_HEADERS64 *nt;
    IMAGE_DOS_HEADER *dos;
    uint8_t *image;

    engine_tls_index = TlsAlloc();

    if (vkd3d_get_env_var("VKD3D_BENCH_DISCARD_KEEP", patches, sizeof(patches)) && atoi(patches) == 1)
        engine_ctl.features |= VKD3D_BENCH_FEATURE_DISCARD_KEEP;
    if (vkd3d_get_env_var("VKD3D_BENCH_COPY_NO_DISCARD", patches, sizeof(patches)))
        engine_ctl.features |= atoi(patches) >= 2 ? VKD3D_BENCH_FEATURE_COPY_KEEP_ALL :
                atoi(patches) == 1 ? VKD3D_BENCH_FEATURE_COPY_KEEP_SYSMEM : 0;

    if (vkd3d_get_env_var("VKD3D_FRAMETRACE", path, sizeof(path)) && *path)
    {
        if ((frametrace_file = fopen(path, "w")))
        {
            setvbuf(frametrace_file, NULL, _IOFBF, 1 << 16);
            frametrace_t0 = vkd3d_get_current_time_ns();
            fprintf(frametrace_file, "# frametrace v1\n");
            vkd3d_frametrace_watching = vkd3d_get_env_var("VKD3D_FRAMETRACE_VA", patches, sizeof(patches)) &&
                    *patches == '1';
            frametrace_detail = vkd3d_get_env_var("VKD3D_FRAMETRACE_DETAIL", patches, sizeof(patches)) &&
                    *patches == '1';
        }
        else
            ERR("frametrace: failed to open %s.\n", path);
    }

    image = (uint8_t *)GetModuleHandleW(NULL);
    dos = (IMAGE_DOS_HEADER *)image;
    nt = (IMAGE_NT_HEADERS64 *)(image + dos->e_lfanew);
    if (nt->FileHeader.TimeDateStamp != ENGINE_EXE_TIMESTAMP || nt->OptionalHeader.SizeOfImage != ENGINE_EXE_SIZE)
    {
        ERR("engine patch: exe timestamp %#x size %#x is not the known build, not patching.\n",
                (unsigned int)nt->FileHeader.TimeDateStamp, (unsigned int)nt->OptionalHeader.SizeOfImage);
        return;
    }

    vkd3d_bench_input_start(image);

    if ((vkd3d_get_env_var("VKD3D_BENCH_INDEX_CACHE", patches, sizeof(patches)) && patches[0] == '1') ||
            vkd3d_geomirror_hooks_wanted())
        engine_patch_geometry_writes(image, nt->OptionalHeader.SizeOfImage);
    if (vkd3d_skinbuf_wanted())
        engine_patch_skin(image, nt->OptionalHeader.SizeOfImage);

    if (!vkd3d_get_env_var("VKD3D_ENGINE_PATCH", patches, sizeof(patches)) || !*patches)
        return;

    if (vkd3d_debug_list_has_member(patches, "spin"))
        engine_patch_spin(image, nt->OptionalHeader.SizeOfImage);
    if (vkd3d_debug_list_has_member(patches, "lgcap"))
        engine_patch_lightgrid_cap(image);
    if (vkd3d_debug_list_has_member(patches, "gdec"))
        engine_patch_gdec(image, nt->OptionalHeader.SizeOfImage);
    if (vkd3d_debug_list_has_member(patches, "dropdst"))
        engine_patch_dropdst(image, nt->OptionalHeader.SizeOfImage);
    if (vkd3d_debug_list_has_member(patches, "qasync"))
    {
        engine_qasync = true;
        engine_ctl.features |= VKD3D_BENCH_FEATURE_QASYNC;
    }
    if (vkd3d_debug_list_has_member(patches, "qdefer") || engine_qasync)
        engine_patch_qdefer(image, nt->OptionalHeader.SizeOfImage);
}

void vkd3d_engine_patch_apply(void)
{
    pthread_once(&engine_patch_once, engine_patch_init);
}

/* Caller holds frametrace_lock. */
static void frametrace_flush_run(struct frametrace_rt *run)
{
    unsigned int i;

    if (!run->draws)
        return;
    for (i = 0; i < frametrace_rt_count; i++)
    {
        if (frametrace_rts[i].resource == run->resource && frametrace_rts[i].layer == run->layer)
        {
            frametrace_rts[i].draws += run->draws;
            break;
        }
    }
    if (i == frametrace_rt_count && i < FRAMETRACE_MAX_RTS)
        frametrace_rts[frametrace_rt_count++] = *run;
    run->draws = 0;
}

void vkd3d_frametrace_count_draw(struct d3d12_command_list *list)
{
    const struct d3d12_rtv_desc *rt = NULL;
    const struct d3d12_resource *resource;
    WCHAR name[48];
    struct engine_tls *tls;
    uint32_t layer, size;
    unsigned int i;

    if (!frametrace_file)
        return;
    __atomic_fetch_add(&frametrace_draws, 1, __ATOMIC_RELAXED);
    /* Per-target runs (R lines) only with VKD3D_FRAMETRACE_DETAIL=1: the run bookkeeping (a lock and
     * a private-data name lookup on every target change) cost 3.6% of the render thread. */
    if (!frametrace_detail || !(tls = engine_tls_get()))
        return;

    if (list->rtvs[0].resource)
        rt = &list->rtvs[0];
    else if (list->dsv.resource)
        rt = &list->dsv;
    resource = rt ? rt->resource : NULL;
    layer = rt && rt->view ? rt->view->info.texture.layer_idx : 0;

    if (resource != tls->run.resource || layer != tls->run.layer)
    {
        pthread_mutex_lock(&frametrace_lock);
        frametrace_flush_run(&tls->run);
        pthread_mutex_unlock(&frametrace_lock);

        memset(&tls->run, 0, sizeof(tls->run));
        tls->run.resource = resource;
        tls->run.layer = layer;
        tls->run.dsv = rt == &list->dsv;
        if (resource)
        {
            tls->run.width = resource->desc.Width;
            tls->run.height = resource->desc.Height;
            tls->run.depth = resource->desc.DepthOrArraySize;
            tls->run.format = resource->desc.Format;
            size = sizeof(name) - sizeof(WCHAR);
            memset(name, 0, sizeof(name));
            if (SUCCEEDED(ID3D12Resource_GetPrivateData((ID3D12Resource *)&resource->ID3D12Resource_iface,
                    &WKPDID_D3DDebugObjectNameW, &size, name)))
            {
                for (i = 0; i + 1 < sizeof(tls->run.name) && name[i]; i++)
                    tls->run.name[i] = name[i] < 128 ? name[i] : '?';
            }
        }
        else
            strcpy(tls->run.name, "(compute or no target)");
    }
    tls->run.draws++;
}

/* E t then free text: timestamped events from the bench's other patches (light grid and skinning
 * growth, losses), on the same clock as the F lines. */
void vkd3d_frametrace_event(const char *fmt, ...)
{
    va_list args;

    if (!frametrace_file)
        return;
    pthread_mutex_lock(&frametrace_lock);
    fprintf(frametrace_file, "E\t%.3f\t", (vkd3d_get_current_time_ns() - frametrace_t0) / 1e6);
    va_start(args, fmt);
    vfprintf(frametrace_file, fmt, args);
    va_end(args);
    fputc('\n', frametrace_file);
    pthread_mutex_unlock(&frametrace_lock);
}

void vkd3d_frametrace_marker(UINT metadata, const void *data, UINT size)
{
    uint32_t hash = 2166136261u;
    unsigned int i;
    char *name;

    if (!frametrace_file || !(name = vkd3d_decode_pix_string(metadata, data, size)))
        return;

    for (i = 0; name[i]; i++)
        hash = (hash ^ (uint8_t)name[i]) * 16777619u;

    pthread_mutex_lock(&frametrace_lock);
    for (i = 0; i < frametrace_marker_count; i++)
    {
        if (frametrace_markers[i].hash == hash)
        {
            frametrace_markers[i].count++;
            break;
        }
    }
    if (i == frametrace_marker_count && i < FRAMETRACE_MAX_MARKERS)
    {
        frametrace_markers[i].hash = hash;
        frametrace_markers[i].count = 1;
        snprintf(frametrace_markers[i].name, sizeof(frametrace_markers[i].name), "%s", name);
        frametrace_marker_count++;
    }
    pthread_mutex_unlock(&frametrace_lock);
    vkd3d_free(name);
}

/* Viewport sizes the game sets, per second: shows the internal render resolution (dynamic
 * resolution would change it between runs and confound fps comparisons). Lock-free table keyed
 * by width << 16 | height. */
#define FRAMETRACE_MAX_VIEWPORTS 64
static struct
{
    uint32_t key;
    uint32_t count;
} frametrace_viewports[FRAMETRACE_MAX_VIEWPORTS];

void vkd3d_frametrace_viewport(float width, float height)
{
    uint32_t key, slot, expected;
    unsigned int i;

    if (!frametrace_file || width < 1.0f || height < 1.0f || width > 65535.0f || height > 65535.0f)
        return;
    key = (uint32_t)width << 16 | (uint32_t)height;
    slot = (key * 2654435761u) >> 26;
    for (i = 0; i < FRAMETRACE_MAX_VIEWPORTS; i++, slot = (slot + 1) % FRAMETRACE_MAX_VIEWPORTS)
    {
        expected = 0;
        if (__atomic_load_n(&frametrace_viewports[slot].key, __ATOMIC_RELAXED) == key ||
                __atomic_compare_exchange_n(&frametrace_viewports[slot].key, &expected, key, false,
                __ATOMIC_RELAXED, __ATOMIC_RELAXED) || expected == key)
        {
            __atomic_fetch_add(&frametrace_viewports[slot].count, 1, __ATOMIC_RELAXED);
            return;
        }
    }
}

/* Caller holds frametrace_lock. VP t then up to 8 "WxH:count" sorted by count. */
static void frametrace_viewport_stats(double t)
{
    uint32_t keys[FRAMETRACE_MAX_VIEWPORTS], counts[FRAMETRACE_MAX_VIEWPORTS], k, c;
    unsigned int i, j, n = 0;

    for (i = 0; i < FRAMETRACE_MAX_VIEWPORTS; i++)
    {
        if (!(c = __atomic_exchange_n(&frametrace_viewports[i].count, 0, __ATOMIC_RELAXED)))
            continue;
        keys[n] = __atomic_load_n(&frametrace_viewports[i].key, __ATOMIC_RELAXED);
        counts[n++] = c;
    }
    for (i = 1; i < n; i++)
    {
        k = keys[i], c = counts[i];
        for (j = i; j && counts[j - 1] < c; j--)
            keys[j] = keys[j - 1], counts[j] = counts[j - 1];
        keys[j] = k, counts[j] = c;
    }
    fprintf(frametrace_file, "VP\t%.3f", t);
    for (i = 0; i < n && i < 8; i++)
        fprintf(frametrace_file, "\t%ux%u:%u", keys[i] >> 16, keys[i] & 0xffff, counts[i]);
    fprintf(frametrace_file, "\n");
}

/* Shader-visible descriptor heaps the game binds, with where their memory lives. A heap whose
 * descriptor buffer is in system memory makes every descriptor fetch cross PCIe. */
#define FRAMETRACE_MAX_HEAPS 32
static struct
{
    struct d3d12_descriptor_heap *heap;
    uint32_t binds;
} frametrace_heaps[FRAMETRACE_MAX_HEAPS];
static unsigned int frametrace_heap_count;

void vkd3d_frametrace_bind_heap(struct d3d12_descriptor_heap *heap)
{
    const VkPhysicalDeviceMemoryProperties *props;
    unsigned int i;

    if (!frametrace_file)
        return;

    pthread_mutex_lock(&frametrace_lock);
    for (i = 0; i < frametrace_heap_count; i++)
    {
        if (frametrace_heaps[i].heap == heap)
            break;
    }
    if (i == frametrace_heap_count && i < FRAMETRACE_MAX_HEAPS)
    {
        props = &heap->device->memory_properties;
        frametrace_heaps[i].heap = heap;
        frametrace_heap_count++;
        /* type, descriptors, descriptor buffer bytes + memory type/flags, aux allocation bytes + type/flags */
        fprintf(frametrace_file, "DH\t%.3f\t%p\t%u\t%u\t%"PRIu64"\t%u\t%#x\t%"PRIu64"\t%u\t%#x\n",
                (vkd3d_get_current_time_ns() - frametrace_t0) / 1e6, (void *)heap, heap->desc.Type,
                heap->desc.NumDescriptors, (uint64_t)heap->descriptor_buffer.device_allocation.size,
                heap->descriptor_buffer.device_allocation.vk_memory_type,
                heap->descriptor_buffer.device_allocation.vk_memory ?
                        props->memoryTypes[heap->descriptor_buffer.device_allocation.vk_memory_type].propertyFlags : 0,
                (uint64_t)heap->device_allocation.size, heap->device_allocation.vk_memory_type,
                heap->device_allocation.vk_memory ?
                        props->memoryTypes[heap->device_allocation.vk_memory_type].propertyFlags : 0);
    }
    if (i < FRAMETRACE_MAX_HEAPS)
        frametrace_heaps[i].binds++;
    pthread_mutex_unlock(&frametrace_lock);
}

/* VKD3D_FRAMETRACE_VA=1: who references the big buffers (>= 16 MiB) and which parts of them.
 * Every GPU VA the game hands to a command list (vertex/index buffer views, root descriptors,
 * copies, ExecuteIndirect arguments) and every buffer view it writes into a descriptor heap is
 * looked up here. Per buffer and second: counts and bytes per kind of reference, and the 64 KiB
 * pages they cover (this second, and ever). Ranges are what the game declares (a vertex buffer
 * view's whole size), so pages are an upper bound of what the GPU actually reads. */
#define FRAMETRACE_MAX_WATCH 64
#define FRAMETRACE_WATCH_MIN (16ull << 20)
#define FRAMETRACE_PAGE_SHIFT 16
/* Views bigger than this cover a whole pool (bindless): counted, but no pages marked. */
#define FRAMETRACE_HUGE_VIEW (64ull << 20)
bool vkd3d_frametrace_watching;
static struct frametrace_watch
{
    struct d3d12_resource *resource;
    VkDeviceAddress va;
    VkDeviceSize size;
    const uint8_t *cpu;
    uint32_t memtype;
    uint32_t page_words;
    uint32_t count[VKD3D_FRAMETRACE_REF_COUNT];
    uint64_t bytes[VKD3D_FRAMETRACE_REF_COUNT];
    uint32_t huge_views;
    uint64_t *pages_sec, *pages_ever;
} frametrace_watch[FRAMETRACE_MAX_WATCH];
static unsigned int frametrace_watch_count;

void vkd3d_frametrace_watch_resource(struct d3d12_resource *resource, bool destroyed)
{
    const struct vkd3d_memory_allocation *mem;
    struct frametrace_watch *w = NULL;
    uint32_t page_words;
    unsigned int i, n;

    if (!vkd3d_frametrace_watching || !frametrace_file || !d3d12_resource_is_buffer(resource) ||
            resource->desc.Width < FRAMETRACE_WATCH_MIN)
        return;

    pthread_mutex_lock(&frametrace_lock);
    n = __atomic_load_n(&frametrace_watch_count, __ATOMIC_ACQUIRE);
    if (destroyed)
    {
        for (i = 0; i < n; i++)
        {
            if (frametrace_watch[i].resource == resource)
            {
                __atomic_store_n(&frametrace_watch[i].resource, NULL, __ATOMIC_RELEASE);
                fprintf(frametrace_file, "VX\t%.3f\t%p\n", (vkd3d_get_current_time_ns() - frametrace_t0) / 1e6,
                        (void *)resource);
            }
        }
        pthread_mutex_unlock(&frametrace_lock);
        return;
    }

    page_words = ((resource->desc.Width >> FRAMETRACE_PAGE_SHIFT) + 64) / 64;
    /* Reuse a dead slot whose page maps are big enough, else take a new one. */
    for (i = 0; i < n && !w; i++)
    {
        if (!frametrace_watch[i].resource && frametrace_watch[i].page_words >= page_words)
            w = &frametrace_watch[i];
    }
    if (!w && n < FRAMETRACE_MAX_WATCH)
    {
        w = &frametrace_watch[n];
        if (!(w->pages_sec = vkd3d_calloc(page_words, sizeof(uint64_t))) ||
                !(w->pages_ever = vkd3d_calloc(page_words, sizeof(uint64_t))))
        {
            vkd3d_free(w->pages_sec);
            w->pages_sec = NULL;
            pthread_mutex_unlock(&frametrace_lock);
            return;
        }
        w->page_words = page_words;
    }
    if (w)
    {
        mem = resource->heap ? &resource->heap->allocation : &resource->mem;
        memset(w->pages_sec, 0, w->page_words * sizeof(uint64_t));
        memset(w->pages_ever, 0, w->page_words * sizeof(uint64_t));
        memset(w->count, 0, sizeof(w->count));
        memset(w->bytes, 0, sizeof(w->bytes));
        w->huge_views = 0;
        w->va = resource->res.va;
        w->size = resource->desc.Width;
        w->memtype = mem->device_allocation.vk_memory_type;
        w->cpu = resource->mem.cpu_address;
        __atomic_store_n(&w->resource, resource, __ATOMIC_RELEASE);
        if (w == &frametrace_watch[n])
            __atomic_store_n(&frametrace_watch_count, n + 1, __ATOMIC_RELEASE);
        /* resource, VA, bytes, memory type, heap type, resource flags */
        fprintf(frametrace_file, "VW\t%.3f\t%p\t%#"PRIx64"\t%"PRIu64"\t%u\t%u\t%#x\n",
                (vkd3d_get_current_time_ns() - frametrace_t0) / 1e6, (void *)resource, (uint64_t)w->va,
                (uint64_t)w->size, w->memtype,
                resource->heap ? resource->heap->desc.Properties.Type : resource->heap_properties.Type,
                resource->desc.Flags);
    }
    pthread_mutex_unlock(&frametrace_lock);
}

static void frametrace_mark_pages(struct frametrace_watch *w, VkDeviceSize offset, VkDeviceSize size)
{
    uint64_t first = offset >> FRAMETRACE_PAGE_SHIFT, last, p, bit;

    last = size ? min(offset + size - 1, w->size - 1) >> FRAMETRACE_PAGE_SHIFT : first;
    for (p = first; p <= last; p++)
    {
        bit = 1ull << (p & 63);
        if (!(__atomic_load_n(&w->pages_sec[p >> 6], __ATOMIC_RELAXED) & bit))
            __atomic_fetch_or(&w->pages_sec[p >> 6], bit, __ATOMIC_RELAXED);
        if (!(__atomic_load_n(&w->pages_ever[p >> 6], __ATOMIC_RELAXED) & bit))
            __atomic_fetch_or(&w->pages_ever[p >> 6], bit, __ATOMIC_RELAXED);
    }
}

void vkd3d_frametrace_ref_va(enum vkd3d_frametrace_ref kind, VkDeviceAddress va, VkDeviceSize size)
{
    struct frametrace_watch *w;
    unsigned int i, n;

    if (!vkd3d_frametrace_watching || !va)
        return;

    n = __atomic_load_n(&frametrace_watch_count, __ATOMIC_ACQUIRE);
    for (i = 0; i < n; i++)
    {
        w = &frametrace_watch[i];
        if (!__atomic_load_n(&w->resource, __ATOMIC_ACQUIRE) || va - w->va >= w->size)
            continue;
        __atomic_fetch_add(&w->count[kind], 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&w->bytes[kind], size, __ATOMIC_RELAXED);
        if (kind == VKD3D_FRAMETRACE_REF_VIEW && size > FRAMETRACE_HUGE_VIEW)
            __atomic_fetch_add(&w->huge_views, 1, __ATOMIC_RELAXED);
        /* Index/vertex buffer views declare whole pools; the draws say what is read. */
        else if (kind != VKD3D_FRAMETRACE_REF_VB && kind != VKD3D_FRAMETRACE_REF_IB &&
                kind != VKD3D_FRAMETRACE_REF_INDIRECT_DRAW)
            frametrace_mark_pages(w, va - w->va, size);
        return;
    }
}

static struct frametrace_watch *frametrace_watch_find(VkDeviceAddress va)
{
    struct frametrace_watch *w;
    unsigned int i, n;

    n = __atomic_load_n(&frametrace_watch_count, __ATOMIC_ACQUIRE);
    for (i = 0; i < n; i++)
    {
        w = &frametrace_watch[i];
        if (__atomic_load_n(&w->resource, __ATOMIC_ACQUIRE) && va - w->va < w->size)
            return w;
    }
    return NULL;
}

/* Smallest and largest index of an index range, read once from the buffer's CPU mapping and
 * cached by (buffer, offset, count): meshes in a pool don't change while they are resident. */
#define FRAMETRACE_RANGE_CACHE (1u << 18)
static struct
{
    uint64_t key;
    uint32_t lo, hi;
} *frametrace_ranges;
static pthread_mutex_t frametrace_range_lock = PTHREAD_MUTEX_INITIALIZER;

static bool frametrace_index_range(struct frametrace_watch *w, VkDeviceSize offset, UINT count, bool is_32,
        uint32_t *lo, uint32_t *hi)
{
    uint64_t key, h, i;
    uint32_t v, a = UINT32_MAX, b = 0;
    UINT k;

    if (!w->cpu || !count)
        return false;
    key = ((uint64_t)(w - frametrace_watch) << 58) ^ (offset << 20) ^ ((uint64_t)is_32 << 19) ^ count;
    h = (key * 0x9e3779b97f4a7c15ull) >> 46;
    pthread_mutex_lock(&frametrace_range_lock);
    if (!frametrace_ranges && !(frametrace_ranges = vkd3d_calloc(FRAMETRACE_RANGE_CACHE, sizeof(*frametrace_ranges))))
    {
        pthread_mutex_unlock(&frametrace_range_lock);
        return false;
    }
    for (i = 0; i < 8; i++)
    {
        if (frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].key == key + 1)
        {
            *lo = frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].lo;
            *hi = frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].hi;
            pthread_mutex_unlock(&frametrace_range_lock);
            return true;
        }
    }
    pthread_mutex_unlock(&frametrace_range_lock);

    for (k = 0; k < count; k++)
    {
        v = is_32 ? ((const uint32_t *)(w->cpu + offset))[k] : ((const uint16_t *)(w->cpu + offset))[k];
        a = min(a, v);
        b = max(b, v);
    }

    pthread_mutex_lock(&frametrace_range_lock);
    for (i = 0; i < 8; i++)
    {
        if (!frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].key || i == 7)
        {
            frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].key = key + 1;
            frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].lo = a;
            frametrace_ranges[(h + i) & (FRAMETRACE_RANGE_CACHE - 1)].hi = b;
            break;
        }
    }
    pthread_mutex_unlock(&frametrace_range_lock);
    *lo = a;
    *hi = b;
    return true;
}

/* An indexed draw reading a watched buffer: the index range it reads, and the vertex range
 * those indices span in each vertex buffer bound from a watched buffer (per-instance streams
 * are counted the same way, which overstates them a little). */
void vkd3d_frametrace_draw_indexed(struct d3d12_command_list *list, UINT index_count, UINT start_index,
        INT base_vertex)
{
    struct frametrace_watch *w, *wv;
    VkDeviceSize offset, bytes, voff, vbytes;
    uint32_t lo, hi, isz;
    unsigned int slot;

    if (!vkd3d_frametrace_watching || !list->frametrace.ib_va || !(w = frametrace_watch_find(list->frametrace.ib_va)))
        return;
    isz = list->frametrace.ib_32 ? 4 : 2;
    offset = list->frametrace.ib_va - w->va + (VkDeviceSize)start_index * isz;
    bytes = (VkDeviceSize)index_count * isz;
    if (offset + bytes > w->size)
        return;
    __atomic_fetch_add(&w->count[VKD3D_FRAMETRACE_REF_DRAW_INDEX], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&w->bytes[VKD3D_FRAMETRACE_REF_DRAW_INDEX], bytes, __ATOMIC_RELAXED);
    frametrace_mark_pages(w, offset, bytes);

    if (!frametrace_index_range(w, offset, index_count, list->frametrace.ib_32, &lo, &hi))
        return;
    for (slot = 0; slot < ARRAY_SIZE(list->frametrace.vb_va); slot++)
    {
        if (!list->frametrace.vb_va[slot] || !list->frametrace.vb_stride[slot] ||
                !(wv = frametrace_watch_find(list->frametrace.vb_va[slot])) || (int64_t)base_vertex + lo < 0)
            continue;
        voff = ((int64_t)base_vertex + lo) * list->frametrace.vb_stride[slot];
        vbytes = (VkDeviceSize)(hi - lo + 1) * list->frametrace.vb_stride[slot];
        if (voff >= list->frametrace.vb_size[slot])
            continue;
        vbytes = min(vbytes, list->frametrace.vb_size[slot] - voff);
        voff += list->frametrace.vb_va[slot] - wv->va;
        __atomic_fetch_add(&wv->count[VKD3D_FRAMETRACE_REF_DRAW_VERTEX], 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&wv->bytes[VKD3D_FRAMETRACE_REF_DRAW_VERTEX], vbytes, __ATOMIC_RELAXED);
        frametrace_mark_pages(wv, voff, vbytes);
    }
}

void vkd3d_frametrace_ref_view(struct d3d12_device *device, struct d3d12_resource *resource,
        DXGI_FORMAT format, UINT64 first_element, UINT num_elements, UINT stride, bool raw)
{
    const struct vkd3d_format *f;

    if (!vkd3d_frametrace_watching || !resource || !d3d12_resource_is_buffer(resource))
        return;
    if (raw)
        stride = 4;
    else if (!stride)
        stride = (f = vkd3d_get_format(device, format, false)) ? f->byte_count : 4;
    vkd3d_frametrace_ref_va(VKD3D_FRAMETRACE_REF_VIEW, resource->res.va + first_element * stride,
            (VkDeviceSize)num_elements * stride);
}

/* Snapshot scan (touch /run/user/1000/vkd3d-scan while VKD3D_FRAMETRACE_VA=1 is on): copy every
 * watched buffer that lives in device-only memory to a readback buffer, 64 MiB at a time, and count
 * its non-zero 64 KiB pages. Heaps are zero-filled at creation, so a zero page was never written
 * (or only with zeros). BS t resource bytes nonzero-pages highest-nonzero-page+1 */
#define FRAMETRACE_SCAN_CHUNK (64ull << 20)
static struct vkd3d_memory_allocation frametrace_readback;
static bool frametrace_readback_ok;

static void frametrace_scan(void)
{
    const VkPhysicalDeviceMemoryProperties *props;
    struct vkd3d_allocate_heap_memory_info alloc_info;
    struct frametrace_watch *w;
    struct d3d12_device *device;
    VkDeviceSize chunk, len, page;
    unsigned int i, n, nonzero, highest;
    const uint64_t *p;
    uint64_t value;
    size_t k;

    n = __atomic_load_n(&frametrace_watch_count, __ATOMIC_ACQUIRE);
    for (i = 0; i < n; i++)
    {
        w = &frametrace_watch[i];
        if (!w->resource)
            continue;
        device = w->resource->device;
        props = &device->memory_properties;
        if (props->memoryTypes[w->memtype].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            continue;
        if (!frametrace_readback_ok)
        {
            memset(&alloc_info, 0, sizeof(alloc_info));
            alloc_info.heap_desc.Properties.Type = D3D12_HEAP_TYPE_READBACK;
            alloc_info.heap_desc.SizeInBytes = FRAMETRACE_SCAN_CHUNK;
            alloc_info.heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            alloc_info.heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
            alloc_info.extra_allocation_flags = VKD3D_ALLOCATION_FLAG_INTERNAL_SCRATCH;
            if (FAILED(vkd3d_allocate_heap_memory(device, &device->memory_allocator, &alloc_info,
                    &frametrace_readback)) || !frametrace_readback.cpu_address)
                return;
            frametrace_readback_ok = true;
        }
        nonzero = highest = 0;
        for (chunk = 0; chunk < w->size; chunk += FRAMETRACE_SCAN_CHUNK)
        {
            len = min(FRAMETRACE_SCAN_CHUNK, w->size - chunk);
            value = vkd3d_memory_transfer_queue_copy_buffer(&device->memory_transfers,
                    w->resource->res.vk_buffer, w->resource->mem.offset + chunk,
                    frametrace_readback.resource.vk_buffer, frametrace_readback.offset, len, true);
            vkd3d_memory_transfer_queue_wait_value(&device->memory_transfers, value);
            for (page = 0; page < len; page += 1ull << FRAMETRACE_PAGE_SHIFT)
            {
                p = (const uint64_t *)((const uint8_t *)frametrace_readback.cpu_address + page);
                for (k = 0; k < min(1ull << FRAMETRACE_PAGE_SHIFT, len - page) / 8; k++)
                {
                    if (p[k])
                    {
                        nonzero++;
                        highest = (chunk + page) >> FRAMETRACE_PAGE_SHIFT;
                        highest++;
                        break;
                    }
                }
            }
        }
        fprintf(frametrace_file, "BS\t%.3f\t%p\t%"PRIu64"\t%u\t%u\n",
                (vkd3d_get_current_time_ns() - frametrace_t0) / 1e6, (void *)w->resource,
                (uint64_t)w->size, nonzero, highest);
    }
    fflush(frametrace_file);
}

static unsigned int frametrace_popcount(const uint64_t *words, uint32_t n, bool clear)
{
    unsigned int total = 0, i;

    for (i = 0; i < n; i++)
    {
        total += __builtin_popcountll(clear ? __atomic_exchange_n((uint64_t *)&words[i], 0, __ATOMIC_RELAXED)
                : __atomic_load_n(&words[i], __ATOMIC_RELAXED));
    }
    return total;
}

static void frametrace_watch_stats(double t)
{
    uint32_t count[VKD3D_FRAMETRACE_REF_COUNT];
    uint64_t bytes[VKD3D_FRAMETRACE_REF_COUNT];
    struct frametrace_watch *w;
    unsigned int i, k, n, any;

    n = __atomic_load_n(&frametrace_watch_count, __ATOMIC_ACQUIRE);
    for (i = 0; i < n; i++)
    {
        w = &frametrace_watch[i];
        if (!w->resource)
            continue;
        for (k = any = 0; k < VKD3D_FRAMETRACE_REF_COUNT; k++)
        {
            count[k] = __atomic_exchange_n(&w->count[k], 0, __ATOMIC_RELAXED);
            bytes[k] = __atomic_exchange_n(&w->bytes[k], 0, __ATOMIC_RELAXED);
            any |= count[k];
        }
        if (!any)
            continue;
        /* VR t resource memtype, then count:KiB per kind (vb ib root view csrc cdst indirect
         * draw-index draw-vertex indirect-indexed-draws), huge views, pages this second, pages ever
         * (64 KiB each) */
        fprintf(frametrace_file, "VR\t%.3f\t%p\t%u", t, (void *)w->resource, w->memtype);
        for (k = 0; k < VKD3D_FRAMETRACE_REF_COUNT; k++)
            fprintf(frametrace_file, "\t%u:%"PRIu64, count[k], bytes[k] >> 10);
        fprintf(frametrace_file, "\t%u\t%u\t%u\n", __atomic_exchange_n(&w->huge_views, 0, __ATOMIC_RELAXED),
                frametrace_popcount(w->pages_sec, w->page_words, true),
                frametrace_popcount(w->pages_ever, w->page_words, false));
    }
}

void vkd3d_frametrace_present(void)
{
    struct engine_tls *tls;
    uint64_t now, wait_ns = 0;
    unsigned int i;
    uint32_t draws;

    if (!frametrace_file)
        return;

    if ((tls = engine_tls_get()))
    {
        wait_ns = tls->wait_ns;
        tls->wait_ns = 0;
    }

    now = vkd3d_get_current_time_ns();
    draws = __atomic_exchange_n(&frametrace_draws, 0, __ATOMIC_RELAXED);
    pthread_mutex_lock(&frametrace_lock);
    fprintf(frametrace_file, "F\t%.3f\t%.3f\t%u\n", (now - frametrace_t0) / 1e6, wait_ns / 1e6, draws);
    if (tls)
        frametrace_flush_run(&tls->run);
    if (++frametrace_frames > 100 && (draws > 1.6 * frametrace_draws_avg || frametrace_frames % 500 == 0))
    {
        for (i = 0; i < frametrace_marker_count; i++)
            fprintf(frametrace_file, "M\t%.3f\t%u\t%s\n", (now - frametrace_t0) / 1e6,
                    frametrace_markers[i].count, frametrace_markers[i].name);
        for (i = 0; i < frametrace_rt_count; i++)
            fprintf(frametrace_file, "R\t%.3f\t%u\t%ux%ux%u\t%u\t%u\t%s\t%s\t%p\n", (now - frametrace_t0) / 1e6,
                    frametrace_rts[i].draws, frametrace_rts[i].width, frametrace_rts[i].height,
                    frametrace_rts[i].depth, frametrace_rts[i].format, frametrace_rts[i].layer,
                    frametrace_rts[i].dsv ? "ds" : "rt", frametrace_rts[i].name, (const void *)frametrace_rts[i].resource);
    }
    frametrace_marker_count = 0;
    frametrace_rt_count = 0;
    frametrace_draws_avg = frametrace_frames == 1 ? draws : 0.98 * frametrace_draws_avg + 0.02 * draws;
    if (now - frametrace_last_stats > 1000000000ull)
    {
        fprintf(frametrace_file, "S\t%.3f\t%"PRId64"\t%.3f\t%"PRId64"\t%.3f\n", (now - frametrace_t0) / 1e6,
                (int64_t)engine_ctl.waits, engine_ctl.blocked_ns / 1e6, (int64_t)engine_ctl.timeouts,
                engine_ctl.spin_ns / 1e6);
        /* W t then per patched site with time this second: "rva:count:ms" (blocked or spinning) */
        if (engine_image)
        {
            fprintf(frametrace_file, "W\t%.3f", (now - frametrace_t0) / 1e6);
            for (i = 0; i < ARRAY_SIZE(engine_spin_sites); i++)
            {
                uint64_t ns = __atomic_exchange_n(&engine_site_ns[i], 0, __ATOMIC_RELAXED);
                uint64_t n = __atomic_exchange_n(&engine_site_count[i], 0, __ATOMIC_RELAXED);
                if (n)
                    fprintf(frametrace_file, "\t%#x:%"PRIu64":%.2f", engine_spin_sites[i].rva, n, ns / 1e6);
            }
            fprintf(frametrace_file, "\n");
        }
        /* Q t batches ready_at_finalize deferred reissued wait_ms foreign_thread process_hooks pending_now
         * (qdefer/qasync, this second) */
        if (engine_qdefer_active)
        {
            fprintf(frametrace_file, "Q\t%.3f\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%.3f\t%"PRIu64"\t%"PRIu64"\t%u\n",
                    (now - frametrace_t0) / 1e6, engine_qdefer_stat_batches, engine_qdefer_stat_ready,
                    engine_qdefer_stat_deferred, engine_qdefer_stat_reissued, engine_qdefer_stat_wait_ns / 1e6,
                    engine_qdefer_stat_foreign, engine_qdefer_stat_process, engine_qdefer_count);
            engine_qdefer_stat_process = 0;
            engine_qdefer_stat_batches = engine_qdefer_stat_ready = engine_qdefer_stat_deferred = 0;
            engine_qdefer_stat_reissued = engine_qdefer_stat_wait_ns = engine_qdefer_stat_foreign = 0;
        }
        frametrace_viewport_stats((now - frametrace_t0) / 1e6);
        {
            static int heapscan = -1;
            static uint64_t heapscan_last;
            char env[8];

            if (heapscan < 0)
                heapscan = vkd3d_get_env_var("VKD3D_BENCH_HEAPSCAN", env, sizeof(env)) && atoi(env) == 1;
            if (heapscan && now - heapscan_last >= 30000000000ull)
            {
                heapscan_last = now;
                vkd3d_bench_heap_scan();
            }
        }
        for (i = 0; i < frametrace_heap_count; i++)
        {
            if (frametrace_heaps[i].binds)
                fprintf(frametrace_file, "DB\t%.3f\t%p\t%u\n", (now - frametrace_t0) / 1e6,
                        (void *)frametrace_heaps[i].heap, frametrace_heaps[i].binds);
            frametrace_heaps[i].binds = 0;
        }
        if (vkd3d_frametrace_watching)
        {
            frametrace_watch_stats((now - frametrace_t0) / 1e6);
            if (GetFileAttributesA("Z:\\run\\user\\1000\\vkd3d-scan") != INVALID_FILE_ATTRIBUTES)
            {
                DeleteFileA("Z:\\run\\user\\1000\\vkd3d-scan");
                frametrace_scan();
            }
        }
        {
            uint64_t hits, misses, full, bytes, stale, invalidations, verified, mismatched, scan_steps;
            uint32_t used;

            /* IC t draws served from the VRAM index cache, misses (runs copied), draws not cached
             * because it is full, KiB copied, cache ring position MiB, cached runs found written
             * since their copy, geometry writes reported by the engine hooks, hits compared with the
             * source (VKD3D_BENCH_INDEX_CACHE_VERIFY=1), mismatches so far, ring search steps */
            if (vkd3d_index_cache_take_stats(&hits, &misses, &full, &bytes, &used, &stale, &invalidations,
                    &verified, &mismatched, &scan_steps))
                fprintf(frametrace_file, "IC\t%.3f\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%u\t%"PRIu64"\t%"PRIu64
                        "\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", (now - frametrace_t0) / 1e6, hits, misses, full, bytes >> 10,
                        used, stale, invalidations, verified, mismatched, scan_steps);
        }
        {
            uint32_t age[7];

            /* IA t MiB of the index cache held by runs last drawn 0, 1, 2, 3, 4-7, 8+ presents
             * ago, MiB free */
            uint64_t req[7], fail[7];
            uint32_t gap;

            /* IS t requests then failures for runs of 1, 2, 3-4, 5-8, 9-16, 17-64, 65+ pages, largest
             * evictable gap (pages) seen by a failed search */
            if (vkd3d_index_cache_size_stats(req, fail, &gap))
                fprintf(frametrace_file, "IS\t%.3f\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64
                        "\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%u\n",
                        (now - frametrace_t0) / 1e6, req[0], req[1], req[2], req[3], req[4], req[5], req[6],
                        fail[0], fail[1], fail[2], fail[3], fail[4], fail[5], fail[6], gap);
            if (vkd3d_index_cache_age_stats(age))
                fprintf(frametrace_file, "IA\t%.3f\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n", (now - frametrace_t0) / 1e6,
                        age[0], age[1], age[2], age[3], age[4], age[5], age[6]);
        }
        if (vkd3d_geomirror_active())
        {
            uint64_t bytes, runs, syncs, waits, scan_ns;

            /* GM t KiB copied to the VRAM geometry heap, page runs, syncs, staging waits,
             * ms spent scanning the write watch */
            vkd3d_geomirror_take_stats(&bytes, &runs, &syncs, &waits, &scan_ns);
            fprintf(frametrace_file, "GM\t%.3f\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%.3f\n",
                    (now - frametrace_t0) / 1e6, bytes >> 10, runs, syncs, waits, scan_ns / 1e6);
        }
        fflush(frametrace_file);
        frametrace_last_stats = now;
    }
    pthread_mutex_unlock(&frametrace_lock);
}
