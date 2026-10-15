/*
 * Sizes learned from earlier sessions (miles-bench).
 *
 * The lazy light grid and skinning buffer, and later the other bench features, size themselves from
 * what the game actually used, not from constants tuned on one machine. Each feature raises the
 * values it measures (vkd3d_bench_learned_raise: the largest value seen wins) and reads what earlier
 * sessions left (vkd3d_bench_learned_get). The values belong to one exe build: a file written by
 * another build (another TimeDateStamp or SizeOfImage) is ignored, so a game update starts over.
 *
 * The file is "vkd3d-proton.<program>.learned" in VKD3D_BENCH_LEARNED_PATH, or where vkd3d-proton
 * keeps its pipeline cache (VKD3D_SHADER_CACHE_PATH, else the working directory). A thread writes it
 * (to a temporary file, then renamed over the old one) at most every 10 s while something changed,
 * so the game's own threads never wait for the disk and a session that is killed loses at most 10 s.
 * Lines are "<name> <value>"; unknown names and malformed lines are ignored, and so are values no
 * memory size can have (corrupt or hand-edited files).
 *
 * The values also belong to one GPU: the first device records its vendor, device id and VRAM heap
 * size ("gpu" line), and a file written on another GPU is dropped for later sessions (this session
 * may already have read it). Experiments that force the VRAM reserve or the budget hold
 * (VKD3D_BENCH_VRAM_RESERVE_MB, _TEXHEAPS, _CHUNKS, VKD3D_BENCH_BUDGET_HOLD_MB) measure their own
 * forced setup, so they never write the reserve or the hold.
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#include <stdio.h>

static const char *const learned_names[VKD3D_BENCH_LEARNED_COUNT] =
{
    [VKD3D_BENCH_LEARNED_LIGHTGRID_PAGES] = "lightgrid.pages",
    [VKD3D_BENCH_LEARNED_SKIN_BYTES] = "skin.bytes",
    [VKD3D_BENCH_LEARNED_RESERVE_BYTES] = "reserve.bytes",
    [VKD3D_BENCH_LEARNED_HOLD_BYTES] = "hold.bytes",
};

static pthread_once_t learned_once = PTHREAD_ONCE_INIT;
static uint64_t learned_value[VKD3D_BENCH_LEARNED_COUNT];
static bool learned_known[VKD3D_BENCH_LEARNED_COUNT];
static uint32_t learned_timestamp, learned_image_size;
static char learned_path[VKD3D_PATH_MAX];
static bool learned_dirty;
/* "gpu <vendor> <device> <VRAM heap bytes>": from the file, then from the first device. */
static uint32_t learned_gpu_vendor, learned_gpu_device;
static uint64_t learned_gpu_vram;
static bool learned_gpu_known, learned_gpu_seen, learned_vram_forced;

/* No memory size this file tracks comes near 1 TiB: anything above is a corrupt value. */
#define LEARNED_MAX_VALUE (1ull << 40)

static void learned_save(void)
{
    char tmp[VKD3D_PATH_MAX + 8];
    unsigned int i;
    FILE *f;

    snprintf(tmp, sizeof(tmp), "%s.tmp", learned_path);
    if (!(f = fopen(tmp, "w")))
    {
        ERR("bench: cannot write %s.\n", tmp);
        return;
    }
    fprintf(f, "build %#x %#x\n", (unsigned int)learned_timestamp, (unsigned int)learned_image_size);
    if (learned_gpu_known)
        fprintf(f, "gpu %#x %#x %"PRIu64"\n", learned_gpu_vendor, learned_gpu_device, learned_gpu_vram);
    for (i = 0; i < VKD3D_BENCH_LEARNED_COUNT; i++)
    {
        if (learned_known[i])
            fprintf(f, "%s %llu\n", learned_names[i], (unsigned long long)learned_value[i]);
    }
    if (fclose(f) || !MoveFileExA(tmp, learned_path, MOVEFILE_REPLACE_EXISTING))
        ERR("bench: cannot replace %s.\n", learned_path);
}

static DWORD WINAPI learned_thread(void *arg)
{
    for (;;)
    {
        Sleep(10000);
        if (__atomic_exchange_n(&learned_dirty, false, __ATOMIC_ACQ_REL))
            learned_save();
    }
    return 0;
}

static void learned_init(void)
{
    static const char *const forcing[] = {"VKD3D_BENCH_VRAM_RESERVE_MB", "VKD3D_BENCH_VRAM_RESERVE_TEXHEAPS",
            "VKD3D_BENCH_VRAM_RESERVE_CHUNKS", "VKD3D_BENCH_BUDGET_HOLD_MB"};
    char dir[VKD3D_PATH_MAX], program[VKD3D_PATH_MAX], line[256], env[32], *value, *end;
    HMODULE module;
    const IMAGE_NT_HEADERS64 *nt;
    const IMAGE_DOS_HEADER *dos;
    const uint8_t *image;
    uint64_t number;
    bool same = false;
    unsigned int i;
    size_t n;
    FILE *f;

    if (!(image = (const uint8_t *)GetModuleHandleW(NULL)))
        return;
    dos = (const IMAGE_DOS_HEADER *)image;
    nt = (const IMAGE_NT_HEADERS64 *)(image + dos->e_lfanew);
    learned_timestamp = nt->FileHeader.TimeDateStamp;
    learned_image_size = nt->OptionalHeader.SizeOfImage;

    if (!vkd3d_get_env_var("VKD3D_BENCH_LEARNED_PATH", dir, sizeof(dir)) || !*dir)
        vkd3d_get_env_var("VKD3D_SHADER_CACHE_PATH", dir, sizeof(dir));
    for (n = strlen(dir); n > 1 && (dir[n - 1] == '/' || dir[n - 1] == '\\'); n--)
        dir[n - 1] = 0;
    if (!vkd3d_get_program_name(program))
        strcpy(program, "app");
    /* Unix paths go through Wine's Z: drive, with back-slashes (see cache.c on Wine's renames). */
    if (dir[0] == '/')
        snprintf(learned_path, sizeof(learned_path), "Z:%s/vkd3d-proton.%s.learned", dir, program);
    else if (dir[0])
        snprintf(learned_path, sizeof(learned_path), "%s/vkd3d-proton.%s.learned", dir, program);
    else
        snprintf(learned_path, sizeof(learned_path), "vkd3d-proton.%s.learned", program);
    for (i = 0, n = strlen(learned_path); i < n; i++)
    {
        if (learned_path[i] == '/')
            learned_path[i] = '\\';
    }

    if ((f = fopen(learned_path, "r")))
    {
        while (fgets(line, sizeof(line), f))
        {
            if (!(value = strchr(line, ' ')))
                continue;
            *value++ = 0;
            if (!strcmp(line, "build"))
            {
                same = strtoul(value, &end, 0) == learned_timestamp && strtoul(end, NULL, 0) == learned_image_size;
                continue;
            }
            if (!strcmp(line, "gpu"))
            {
                learned_gpu_vendor = strtoul(value, &end, 0);
                learned_gpu_device = strtoul(end, &end, 0);
                learned_gpu_vram = strtoull(end, NULL, 10);
                learned_gpu_known = true;
                continue;
            }
            while (*value == ' ')
                value++;
            number = strtoull(value, &end, 10);
            if (!same || end == value || *value == '-' || number > LEARNED_MAX_VALUE)
                continue;
            for (i = 0; i < VKD3D_BENCH_LEARNED_COUNT; i++)
            {
                if (!strcmp(line, learned_names[i]))
                {
                    learned_value[i] = number;
                    learned_known[i] = true;
                }
            }
        }
        fclose(f);
    }
    for (i = 0; i < VKD3D_BENCH_LEARNED_COUNT; i++)
    {
        if (learned_known[i])
            INFO("bench: learned %s = %"PRIu64" (%s).\n", learned_names[i], learned_value[i], learned_path);
    }
    if (!same)
        INFO("bench: nothing learned yet for this build (%s); features start as the game asks.\n", learned_path);
    for (i = 0; i < ARRAY_SIZE(forcing); i++)
    {
        if (vkd3d_get_env_var(forcing[i], env, sizeof(env)) && *env)
        {
            learned_vram_forced = true;
            INFO("bench: %s is set; this session does not write the learned reserve or hold.\n", forcing[i]);
        }
    }
    /* The writer thread runs code in this DLL for the life of the process: keep the DLL loaded. */
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            (const WCHAR *)learned_thread, &module);
    CloseHandle(CreateThread(NULL, 0, learned_thread, NULL, 0, NULL));
}

/* What earlier sessions of this build measured; false if nothing yet. */
bool vkd3d_bench_learned_get(enum vkd3d_bench_learned key, uint64_t *value)
{
    pthread_once(&learned_once, learned_init);
    if (!learned_known[key])
        return false;
    *value = learned_value[key];
    return true;
}

/* A value seen in this session; kept if it is the largest so far. Cheap unless it is. */
void vkd3d_bench_learned_raise(enum vkd3d_bench_learned key, uint64_t value)
{
    uint64_t old;

    pthread_once(&learned_once, learned_init);
    if (learned_vram_forced && (key == VKD3D_BENCH_LEARNED_RESERVE_BYTES || key == VKD3D_BENCH_LEARNED_HOLD_BYTES))
        return;
    if (value > LEARNED_MAX_VALUE)
        return;
    old = __atomic_load_n(&learned_value[key], __ATOMIC_RELAXED);
    if (__atomic_load_n(&learned_known[key], __ATOMIC_ACQUIRE) && value <= old)
        return;
    /* Unknown values are 0, so "larger than what is there" is right for the first raise too: two
     * threads raising a key at once can only leave the larger value. */
    while (value > old &&
            !__atomic_compare_exchange_n(&learned_value[key], &old, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
        ;
    __atomic_store_n(&learned_known[key], true, __ATOMIC_RELEASE);
    __atomic_store_n(&learned_dirty, true, __ATOMIC_RELEASE);
}

/* The device the values are for. Values learned on another GPU (or with another amount of VRAM)
 * are dropped from the file; the features of this session may already have read them. */
void vkd3d_bench_learned_device(uint32_t vendor, uint32_t device, uint64_t vram)
{
    unsigned int i;

    pthread_once(&learned_once, learned_init);
    if (__atomic_exchange_n(&learned_gpu_seen, true, __ATOMIC_ACQ_REL))
        return;
    if (learned_gpu_known && (learned_gpu_vendor != vendor || learned_gpu_device != device || learned_gpu_vram != vram))
    {
        INFO("bench: %s was learned on GPU %04x:%04x with %"PRIu64" MiB of VRAM, this is %04x:%04x with %"PRIu64
                " MiB; starting over.\n", learned_path, learned_gpu_vendor, learned_gpu_device,
                learned_gpu_vram >> 20, vendor, device, vram >> 20);
        /* Values too, or a smaller raise later would bring the old GPU's value back. */
        for (i = 0; i < VKD3D_BENCH_LEARNED_COUNT; i++)
        {
            __atomic_store_n(&learned_known[i], false, __ATOMIC_RELEASE);
            __atomic_store_n(&learned_value[i], 0, __ATOMIC_RELEASE);
        }
    }
    /* A reserve or hold larger than VRAM is corrupt, not learned. */
    for (i = VKD3D_BENCH_LEARNED_RESERVE_BYTES; vram && i <= VKD3D_BENCH_LEARNED_HOLD_BYTES; i++)
    {
        if (__atomic_load_n(&learned_known[i], __ATOMIC_ACQUIRE) && learned_value[i] >= vram)
        {
            INFO("bench: ignoring learned %s = %"PRIu64": more than this GPU's %"PRIu64" MiB of VRAM.\n",
                    learned_names[i], learned_value[i], vram >> 20);
            __atomic_store_n(&learned_known[i], false, __ATOMIC_RELEASE);
            __atomic_store_n(&learned_value[i], 0, __ATOMIC_RELEASE);
        }
    }
    learned_gpu_vendor = vendor;
    learned_gpu_device = device;
    learned_gpu_vram = vram;
    learned_gpu_known = true;
    __atomic_store_n(&learned_dirty, true, __ATOMIC_RELEASE);
}
