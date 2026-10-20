/*
 * Bench input: drive Miles Morales' launcher and menus from inside the process, so the bench never
 * needs the game window to have keyboard focus (focusing it would pull the user's view to it).
 *
 * VKD3D_BENCH_INPUT=1 starts a thread that polls /run/user/1000/vkd3d-input (inside the game's
 * container: /proc/<pid>/root/run/user/1000/vkd3d-input). Writing a key name there ("enter",
 * "escape", "space", "up", "down", "left", "right", "shift", "ctrl", "tab", "alt", or any single letter or
 * digit) delivers that key,
 * and "hold <ms> <key>..." keeps keys down together (game window only); "pad <ms> <control>..." holds a
 * virtual XInput controller's controls instead (see bench_input_pad):
 *
 *  - to the launcher window (class GameNxApp, wndproc 0x21ea670) as posted WM_KEYDOWN/WM_KEYUP; its
 *    window procedure acts on WM_KEYDOWN directly, without looking at focus;
 *  - to the game window (wndproc 0x21dfc80, wrapped by 0x35fe760) as posted WM_INPUT messages whose HRAWINPUT is a tagged
 *    value. The game reads keyboard input only through WM_INPUT + GetRawInputData, so its import
 *    slot for GetRawInputData is pointed at a wrapper that answers for tagged handles with a
 *    keyboard RAWINPUT and passes everything else through.
 *
 * Each command is answered in vkd3d-input.done ("launcher" / "game" / "none", plus the window
 * object's input-gate function for RE).
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#include <ctype.h>
#include <stdio.h>

#define BENCH_INPUT_FILE "Z:\\run\\user\\1000\\vkd3d-input"
#define BENCH_INPUT_DONE "Z:\\run\\user\\1000\\vkd3d-input.done"

/* MilesMorales.exe v1.1116.0.0 (TimeDateStamp 0x6374c7f4). */
#define BENCH_RVA_LAUNCHER_WNDPROC 0x21ea670u
#define BENCH_RVA_GAME_WNDPROC 0x21dfc80u
#define BENCH_RVA_GAME_WNDPROC_WRAPPER 0x35fe760u /* installed over it; filters WM_DEVICECHANGE */
#define BENCH_RVA_IAT_GETRAWINPUTDATA 0x3d19ca0u
#define BENCH_RVA_NXAPP 0x84aed60u /* NxApp *, whose +0x98 is the object the game wndproc asks first */

/* Tagged HRAWINPUT: magic in the high bits, virtual key << 1, bit 0 = key up. */
#define BENCH_INPUT_MAGIC 0x6d620000u
#define BENCH_INPUT_MAGIC_MASK 0xfffffe00u

typedef UINT (WINAPI *PFN_GetRawInputData)(HRAWINPUT, UINT, void *, UINT *, UINT);

static struct
{
    uint8_t *image;
    HMODULE user32;
    PFN_GetRawInputData real_get_raw_input_data;
    BOOL (WINAPI *post_message)(HWND, UINT, WPARAM, LPARAM);
    BOOL (WINAPI *enum_windows)(WNDENUMPROC, LPARAM);
    DWORD (WINAPI *get_window_thread_process_id)(HWND, DWORD *);
    LONG_PTR (WINAPI *get_window_long_ptr)(HWND, int);
    UINT (WINAPI *map_virtual_key)(UINT, UINT);
    UINT (WINAPI *get_raw_input_device_list)(RAWINPUTDEVICELIST *, UINT *, UINT);
    int (WINAPI *get_class_name)(HWND, char *, int);
    BOOL (WINAPI *is_window_visible)(HWND);
    BOOL (WINAPI *get_window_rect)(HWND, RECT *);
    HANDLE keyboard;
    bool hooked;
} bench_input;

static UINT WINAPI bench_get_raw_input_data(HRAWINPUT handle, UINT command, void *data, UINT *size, UINT header_size)
{
    uintptr_t h = (uintptr_t)handle;
    UINT vk, n;
    RAWINPUT ri;

    if ((h & BENCH_INPUT_MAGIC_MASK) != BENCH_INPUT_MAGIC)
        return bench_input.real_get_raw_input_data(handle, command, data, size, header_size);

    vk = (h >> 1) & 0xff;
    memset(&ri, 0, sizeof(ri));
    ri.header.dwType = RIM_TYPEKEYBOARD;
    ri.header.dwSize = sizeof(RAWINPUTHEADER) + sizeof(RAWKEYBOARD);
    ri.header.hDevice = bench_input.keyboard;
    ri.header.wParam = RIM_INPUT;
    ri.data.keyboard.MakeCode = bench_input.map_virtual_key(vk, 0 /* MAPVK_VK_TO_VSC */);
    ri.data.keyboard.Flags = (h & 1) ? RI_KEY_BREAK : RI_KEY_MAKE;
    if (vk == VK_UP || vk == VK_DOWN || vk == VK_LEFT || vk == VK_RIGHT)
        ri.data.keyboard.Flags |= RI_KEY_E0;
    ri.data.keyboard.VKey = vk;
    ri.data.keyboard.Message = (h & 1) ? WM_KEYUP : WM_KEYDOWN;

    n = command == RID_HEADER ? sizeof(RAWINPUTHEADER) : ri.header.dwSize;
    if (!data)
    {
        *size = n;
        return 0;
    }
    if (*size < n)
    {
        *size = n;
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return (UINT)-1;
    }
    memcpy(data, &ri, n);
    return n;
}

/* Virtual XInput controller, user index 0. The game loads XInput at run time (XInput1_4, 1_3 or
 * 9_1_0, through GetProcAddress), so the loaded DLL's XInputGetState and XInputGetCapabilities are
 * overwritten with a jump to these on the first "pad" command. Bench processes only (cfgrun.sh sets
 * VKD3D_BENCH_INPUT); no real controller is passed through. */
struct bench_xinput_gamepad
{
    WORD buttons;
    BYTE left_trigger, right_trigger;
    SHORT thumb_lx, thumb_ly, thumb_rx, thumb_ry;
};

struct bench_xinput_state
{
    DWORD packet;
    struct bench_xinput_gamepad pad;
};

struct bench_xinput_capabilities
{
    BYTE type, sub_type;
    WORD flags;
    struct bench_xinput_gamepad pad;
    WORD left_motor, right_motor;
};

static struct
{
    struct bench_xinput_state state;
    bool patched;
    LONG calls;
} bench_pad;

static DWORD WINAPI bench_xinput_get_state(DWORD user, struct bench_xinput_state *state)
{
    InterlockedIncrement(&bench_pad.calls);
    if (user)
        return ERROR_DEVICE_NOT_CONNECTED;
    *state = bench_pad.state;
    return ERROR_SUCCESS;
}

static DWORD WINAPI bench_xinput_get_capabilities(DWORD user, DWORD flags, struct bench_xinput_capabilities *caps)
{
    if (user)
        return ERROR_DEVICE_NOT_CONNECTED;
    memset(caps, 0, sizeof(*caps));
    caps->type = 1; /* XINPUT_DEVTYPE_GAMEPAD */
    caps->sub_type = 1; /* XINPUT_DEVSUBTYPE_GAMEPAD */
    caps->pad.buttons = 0xf3ff;
    caps->pad.left_trigger = caps->pad.right_trigger = 0xff;
    caps->pad.thumb_lx = caps->pad.thumb_ly = caps->pad.thumb_rx = caps->pad.thumb_ry = (SHORT)0xffc0;
    caps->left_motor = caps->right_motor = 0xff;
    return ERROR_SUCCESS;
}

/* mov rax, imm64; jmp rax */
static void bench_input_patch_function(uint8_t *code, void *target)
{
    uint8_t jump[12] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0};
    DWORD old_protect;

    memcpy(jump + 2, &target, sizeof(target));
    VirtualProtect(code, sizeof(jump), PAGE_EXECUTE_READWRITE, &old_protect);
    memcpy(code, jump, sizeof(jump));
    VirtualProtect(code, sizeof(jump), old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), code, sizeof(jump));
}

static unsigned int bench_input_patch_xinput(void)
{
    static const char *dlls[] = {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"};
    unsigned int i, patched = 0;
    HMODULE module;
    void *f;

    for (i = 0; i < ARRAY_SIZE(dlls); i++)
    {
        if (!(module = GetModuleHandleA(dlls[i])))
            continue;
        if ((f = (void *)GetProcAddress(module, "XInputGetState")))
        {
            bench_input_patch_function(f, (void *)bench_xinput_get_state);
            patched++;
        }
        /* Ordinal 100, XInputGetStateEx (the same state plus the guide button), where exported. */
        if ((f = (void *)GetProcAddress(module, (const char *)(uintptr_t)100)))
            bench_input_patch_function(f, (void *)bench_xinput_get_state);
        if ((f = (void *)GetProcAddress(module, "XInputGetCapabilities")))
            bench_input_patch_function(f, (void *)bench_xinput_get_capabilities);
        INFO("bench input: %s patched to the virtual controller.\n", dlls[i]);
    }
    return patched;
}

/* "pad <ms> <control>...": hold the virtual controller's controls for <ms>, then return it to rest
 * (it stays connected). Controls: lx=, ly=, rx=, ry= (stick position -1..1; ly=1 is forward), lt, rt
 * (fully pulled), and the buttons a b x y lb rb start back ls rs up down left right. */
static void bench_input_pad(char *args, FILE *done)
{
    static const struct { const char *name; WORD bit; } buttons[] =
    {
        {"up", 0x0001}, {"down", 0x0002}, {"left", 0x0004}, {"right", 0x0008}, {"start", 0x0010},
        {"back", 0x0020}, {"ls", 0x0040}, {"rs", 0x0080}, {"lb", 0x0100}, {"rb", 0x0200},
        {"a", 0x1000}, {"b", 0x2000}, {"x", 0x4000}, {"y", 0x8000},
    };
    struct bench_xinput_gamepad pad = {0};
    unsigned int i;
    char *tok;
    DWORD ms;

    if (!bench_pad.patched)
    {
        if (!bench_input_patch_xinput())
        {
            fprintf(done, "none pad (no XInput DLL loaded)\n");
            return;
        }
        bench_pad.patched = true;
    }

    ms = strtoul(args, &args, 10);
    while ((tok = strtok(args, " ")))
    {
        args = NULL;
        if (strchr(tok, '='))
        {
            SHORT v = (SHORT)max(-32767.0, min(32767.0, atof(strchr(tok, '=') + 1) * 32767.0));
            if (!strncmp(tok, "lx=", 3)) pad.thumb_lx = v;
            else if (!strncmp(tok, "ly=", 3)) pad.thumb_ly = v;
            else if (!strncmp(tok, "rx=", 3)) pad.thumb_rx = v;
            else if (!strncmp(tok, "ry=", 3)) pad.thumb_ry = v;
            else goto unknown;
            continue;
        }
        if (!strcmp(tok, "lt")) { pad.left_trigger = 0xff; continue; }
        if (!strcmp(tok, "rt")) { pad.right_trigger = 0xff; continue; }
        for (i = 0; i < ARRAY_SIZE(buttons); i++)
            if (!strcmp(tok, buttons[i].name))
                break;
        if (i == ARRAY_SIZE(buttons))
            goto unknown;
        pad.buttons |= buttons[i].bit;
    }

    bench_pad.state.pad = pad;
    bench_pad.state.packet++;
    Sleep(ms);
    memset(&bench_pad.state.pad, 0, sizeof(bench_pad.state.pad));
    bench_pad.state.packet++;
    fprintf(done, "pad %lu ms, %ld XInputGetState calls so far\n", (unsigned long)ms, (long)bench_pad.calls);
    return;

unknown:
    fprintf(done, "unknown pad control %s\n", tok);
}

static bool bench_input_hook(void)
{
    void **slot = (void **)(bench_input.image + BENCH_RVA_IAT_GETRAWINPUTDATA);
    RAWINPUTDEVICELIST devices[32];
    UINT count = ARRAY_SIZE(devices), i;
    DWORD old_protect;

    if (bench_input.hooked)
        return true;

    if (*slot != (void *)GetProcAddress(bench_input.user32, "GetRawInputData"))
    {
        ERR("bench input: GetRawInputData import slot holds %p, not user32's; not hooking.\n", *slot);
        return false;
    }

    if (bench_input.get_raw_input_device_list(devices, &count, sizeof(devices[0])) != (UINT)-1)
    {
        for (i = 0; i < count && !bench_input.keyboard; i++)
            if (devices[i].dwType == RIM_TYPEKEYBOARD)
                bench_input.keyboard = devices[i].hDevice;
    }

    bench_input.real_get_raw_input_data = (PFN_GetRawInputData)*slot;
    VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &old_protect);
    *slot = (void *)bench_get_raw_input_data;
    VirtualProtect(slot, sizeof(*slot), old_protect, &old_protect);
    bench_input.hooked = true;
    INFO("bench input: GetRawInputData hooked, keyboard device %p.\n", bench_input.keyboard);
    return true;
}

struct bench_input_windows
{
    HWND launcher, game, largest;
    LONG largest_area;
    FILE *log;
};

/* The launcher and the game window are told apart by their window procedures; failing that, the
 * game window is the largest visible top-level window of the process that isn't the launcher.
 * Every window of the process is listed in the answer file, so a miss can be read afterwards. */
static BOOL CALLBACK bench_input_enum(HWND hwnd, LPARAM param)
{
    struct bench_input_windows *w = (struct bench_input_windows *)param;
    uintptr_t proc, image = (uintptr_t)bench_input.image;
    char name[64] = "";
    DWORD pid = 0;
    RECT rect = {0};
    LONG area;
    BOOL visible;

    bench_input.get_window_thread_process_id(hwnd, &pid);
    if (pid != GetCurrentProcessId())
        return TRUE;
    proc = (uintptr_t)bench_input.get_window_long_ptr(hwnd, GWLP_WNDPROC);
    bench_input.get_class_name(hwnd, name, sizeof(name));
    visible = bench_input.is_window_visible(hwnd);
    bench_input.get_window_rect(hwnd, &rect);
    area = (rect.right - rect.left) * (rect.bottom - rect.top);
    if (w->log)
        fprintf(w->log, "window %p class %s proc %#"PRIx64"%s visible %d %ldx%ld\n", hwnd, name,
                (uint64_t)(proc - (proc >= image && proc < image + 0x8fef000u ? image : 0)),
                proc >= image && proc < image + 0x8fef000u ? " (exe rva)" : "", visible,
                rect.right - rect.left, rect.bottom - rect.top);
    if (proc == image + BENCH_RVA_LAUNCHER_WNDPROC)
        w->launcher = hwnd;
    else if (proc == image + BENCH_RVA_GAME_WNDPROC || proc == image + BENCH_RVA_GAME_WNDPROC_WRAPPER)
        w->game = hwnd;
    else if (visible && area > w->largest_area)
    {
        w->largest = hwnd;
        w->largest_area = area;
    }
    return TRUE;
}

static UINT bench_input_vk(const char *name)
{
    static const struct { const char *name; UINT vk; } keys[] =
    {
        {"enter", VK_RETURN}, {"escape", VK_ESCAPE}, {"space", VK_SPACE},
        {"up", VK_UP}, {"down", VK_DOWN}, {"left", VK_LEFT}, {"right", VK_RIGHT},
        {"shift", VK_SHIFT}, {"ctrl", VK_CONTROL}, {"tab", VK_TAB}, {"alt", VK_MENU},
    };
    unsigned int i;

    for (i = 0; i < ARRAY_SIZE(keys); i++)
        if (!strcmp(name, keys[i].name))
            return keys[i].vk;
    /* Any single letter or digit: its virtual-key code is its uppercase character. */
    if (name[0] && !name[1] && ((name[0] >= 'a' && name[0] <= 'z') || (name[0] >= '0' && name[0] <= '9')))
        return (UINT)toupper((unsigned char)name[0]);
    return 0;
}

/* The object the game wndproc asks (vtable +0x68) before it reads WM_INPUT; logged so a dropped key
 * can be traced to that gate. */
static uintptr_t bench_input_gate_rva(void)
{
    uintptr_t app, obj, vtbl;

    if (!(app = *(uintptr_t *)(bench_input.image + BENCH_RVA_NXAPP)))
        return 0;
    if (!(obj = *(uintptr_t *)(app + 0x98)) || !(vtbl = *(uintptr_t *)obj))
        return 0;
    return *(uintptr_t *)(vtbl + 0x68) - (uintptr_t)bench_input.image;
}

static void bench_input_send(const char *name, FILE *done)
{
    struct bench_input_windows w = {.log = done};
    LPARAM lparam;
    UINT vk, sc;

    if (!(vk = bench_input_vk(name)))
    {
        fprintf(done, "unknown key %s\n", name);
        return;
    }
    bench_input.enum_windows(bench_input_enum, (LPARAM)&w);
    if (!w.game && !w.launcher)
        w.game = w.largest;
    sc = bench_input.map_virtual_key(vk, 0 /* MAPVK_VK_TO_VSC */);

    if (w.launcher)
    {
        lparam = 1 | (sc << 16);
        bench_input.post_message(w.launcher, WM_KEYDOWN, vk, lparam);
        Sleep(60);
        bench_input.post_message(w.launcher, WM_KEYUP, vk, lparam | (1u << 30) | (1u << 31));
        fprintf(done, "launcher %s\n", name);
        INFO("bench input: %s to launcher %p.\n", name, w.launcher);
    }
    else if (w.game && bench_input_hook())
    {
        bench_input.post_message(w.game, WM_INPUT, RIM_INPUT, (LPARAM)(uintptr_t)(BENCH_INPUT_MAGIC | vk << 1));
        Sleep(80);
        bench_input.post_message(w.game, WM_INPUT, RIM_INPUT, (LPARAM)(uintptr_t)(BENCH_INPUT_MAGIC | vk << 1 | 1));
        fprintf(done, "game %s gate %#"PRIx64"\n", name, (uint64_t)bench_input_gate_rva());
        INFO("bench input: %s to game window %p, input gate rva %#"PRIx64".\n", name, w.game,
                (uint64_t)bench_input_gate_rva());
    }
    else
    {
        fprintf(done, "none %s\n", name);
    }
}

/* "hold <ms> <key> [<key>...]": press the keys together, keep them down for <ms>, release them.
 * Game window only (movement for traversal runs). */
static void bench_input_hold(char *args, FILE *done)
{
    struct bench_input_windows w = {.log = done};
    UINT vks[8], n = 0, i;
    DWORD ms, end;
    char *tok;

    ms = strtoul(args, &args, 10);
    while (*args && n < ARRAY_SIZE(vks))
    {
        while (*args == ' ')
            args++;
        if (!*args)
            break;
        tok = args;
        while (*args && *args != ' ')
            args++;
        if (*args)
            *args++ = '\0';
        if (!(vks[n] = bench_input_vk(tok)))
        {
            fprintf(done, "unknown key %s\n", tok);
            return;
        }
        n++;
    }
    bench_input.enum_windows(bench_input_enum, (LPARAM)&w);
    if (!w.game)
        w.game = w.largest;
    if (!n || !w.game || w.launcher || !bench_input_hook())
    {
        fprintf(done, "none hold\n");
        return;
    }
    /* Key down, repeated every 30 ms while held, as a held key's autorepeat arrives. */
    for (end = GetTickCount() + ms; (LONG)(end - GetTickCount()) > 0; Sleep(30))
        for (i = 0; i < n; i++)
            bench_input.post_message(w.game, WM_INPUT, RIM_INPUT, (LPARAM)(uintptr_t)(BENCH_INPUT_MAGIC | vks[i] << 1));
    for (i = 0; i < n; i++)
        bench_input.post_message(w.game, WM_INPUT, RIM_INPUT, (LPARAM)(uintptr_t)(BENCH_INPUT_MAGIC | vks[i] << 1 | 1));
    fprintf(done, "game hold %u keys %lu ms\n", n, (unsigned long)ms);
    INFO("bench input: held %u keys for %lu ms.\n", n, (unsigned long)ms);
}

static DWORD WINAPI bench_input_thread(void *arg)
{
    char line[256];
    size_t len;
    FILE *f, *done;

    for (;;)
    {
        Sleep(200);
        if (!(f = fopen(BENCH_INPUT_FILE, "r")))
            continue;
        len = fread(line, 1, sizeof(line) - 1, f);
        fclose(f);
        DeleteFileA(BENCH_INPUT_FILE);
        line[len] = '\0';
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' '))
            line[--len] = '\0';
        if (!(done = fopen(BENCH_INPUT_DONE ".tmp", "w")))
            continue;
        if (!strncmp(line, "hold ", 5))
            bench_input_hold(line + 5, done);
        else if (!strncmp(line, "pad ", 4))
            bench_input_pad(line + 4, done);
        else
            bench_input_send(line, done);
        fclose(done);
        MoveFileExA(BENCH_INPUT_DONE ".tmp", BENCH_INPUT_DONE, MOVEFILE_REPLACE_EXISTING);
    }
    return 0;
}

void vkd3d_bench_input_start(uint8_t *image)
{
    char value[8];

    if (!vkd3d_get_env_var("VKD3D_BENCH_INPUT", value, sizeof(value)) || *value != '1')
        return;
    if (!(bench_input.user32 = LoadLibraryA("user32.dll")))
        return;
    bench_input.image = image;
    bench_input.post_message = (void *)GetProcAddress(bench_input.user32, "PostMessageW");
    bench_input.enum_windows = (void *)GetProcAddress(bench_input.user32, "EnumWindows");
    bench_input.get_window_thread_process_id = (void *)GetProcAddress(bench_input.user32, "GetWindowThreadProcessId");
    bench_input.get_window_long_ptr = (void *)GetProcAddress(bench_input.user32, "GetWindowLongPtrW");
    bench_input.map_virtual_key = (void *)GetProcAddress(bench_input.user32, "MapVirtualKeyW");
    bench_input.get_raw_input_device_list = (void *)GetProcAddress(bench_input.user32, "GetRawInputDeviceList");
    bench_input.get_class_name = (void *)GetProcAddress(bench_input.user32, "GetClassNameA");
    bench_input.is_window_visible = (void *)GetProcAddress(bench_input.user32, "IsWindowVisible");
    bench_input.get_window_rect = (void *)GetProcAddress(bench_input.user32, "GetWindowRect");
    CloseHandle(CreateThread(NULL, 0, bench_input_thread, NULL, 0, NULL));
    INFO("bench input: polling %s.\n", BENCH_INPUT_FILE);
}
