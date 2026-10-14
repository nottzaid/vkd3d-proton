/*
 * Bench GPU timeline: where the GPU time of a frame goes, by segment of work between the points
 * where the GPU synchronises anyway.
 *
 * VKD3D_BENCH_GPUTIME=<path> wraps a few Vulkan commands in the device's dispatch table. Every
 * VKD3D_BENCH_GPUTIME_EVERY presents (default 64), the command buffers begun before the next present
 * are instrumented: a timestamp (ALL_COMMANDS, i.e. when everything recorded before it has finished)
 * at the start of the command buffer, after every pipeline barrier, before and after every render
 * pass instance, and at the end. Barriers and render pass boundaries already serialise the GPU, so
 * the timestamps add little. Each segment between two timestamps is labelled with what it holds
 * (draws, dispatches and their workgroups, copies, clears, pipeline binds, render area).
 *
 * Results are read back without waiting, at later presents, one line per segment:
 *   GT frame queue seq ts_ns dur_ns kind draws dispatches groups copies clears binds WxH pipeline
 *      shader_hash XxYxZ (the DXIL hash and workgroup counts of the segment's first direct Dispatch)
 * ts_ns is the segment's end on the GPU clock; dur_ns = end - previous timestamp in the same
 * command buffer (for the first segment, end - start of the command buffer). kind is the mark that
 * ends the segment: b(arrier) r(ender pass begins) e(nd of render pass) z (end of command buffer).
 * Command buffers also get a "GS frame queue seq start_ns submit_cpu_ms" line (CPU time of the
 * vkQueueSubmit2, ms since the Unix epoch like the GPU timer), presents a "GP frame cpu_ns" line.
 * VKD3D_BENCH_GPUTIME_RUN=n samples n consecutive frames each time (default 1).
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

#include <stdio.h>
#include <stdarg.h>

#define GT_MAX_QUERIES 1024
#define GT_MAX_CMDS 256
#define GT_TABLE_SIZE 1024
#define GT_MAX_NOTES 512

struct gt_seg
{
    char kind;
    uint16_t width, height;
    uint32_t draws, dispatches, copies, clears, binds;
    uint64_t groups;
    uint64_t pipeline;
    uint64_t shader; /* DXIL hash of the first dispatch (vkd3d's shader dump name) */
    uint32_t dims[3];
};

struct gt_cmd
{
    VkCommandBuffer cmd;
    VkQueryPool pool;
    uint32_t count;
    uint64_t frame, seq, submit_ns;
    VkQueue queue;
    bool submitted;
    uint32_t rp_width, rp_height;
    struct gt_seg cur;
    struct gt_seg segs[GT_MAX_QUERIES];
    uint32_t note_count;
    struct
    {
        uint32_t at; /* timestamps written before the note: the note belongs to segment <at> */
        char text[124];
    } notes[GT_MAX_NOTES];
    struct gt_cmd *next;
};

static const struct vkd3d_vk_device_procs *gt_vk;
static struct vkd3d_vk_device_procs gt_orig;
static VkDevice gt_device;
static FILE *gt_file;
static double gt_period;
static uint32_t gt_every = 64, gt_run = 1;
static uint64_t gt_presents;
static uint64_t gt_window_frame; /* frame being sampled, 0 when none */
static uint64_t gt_seq;
static uint32_t gt_tracked; /* entries in gt_table */
static pthread_mutex_t gt_lock = PTHREAD_MUTEX_INITIALIZER;
static struct
{
    VkCommandBuffer key;
    struct gt_cmd *cmd;
} gt_table[GT_TABLE_SIZE];
static struct gt_cmd *gt_free, *gt_pending;
static uint32_t gt_allocated;

static uint32_t gt_hash(VkCommandBuffer cmd)
{
    uint64_t v = (uint64_t)(uintptr_t)cmd;
    return (uint32_t)((v ^ (v >> 17)) * 0x9e3779b1u) % GT_TABLE_SIZE;
}

static struct gt_cmd *gt_lookup(VkCommandBuffer cmd)
{
    uint32_t i, slot = gt_hash(cmd);

    if (!__atomic_load_n(&gt_tracked, __ATOMIC_RELAXED))
        return NULL;
    for (i = 0; i < GT_TABLE_SIZE; i++, slot = (slot + 1) % GT_TABLE_SIZE)
    {
        VkCommandBuffer key = __atomic_load_n(&gt_table[slot].key, __ATOMIC_ACQUIRE);
        if (key == cmd)
            return gt_table[slot].cmd;
        if (!key)
            return NULL;
    }
    return NULL;
}

/* Caller holds gt_lock. Linear probing with deletion by backward shift. */
static void gt_table_remove(VkCommandBuffer cmd)
{
    uint32_t i = gt_hash(cmd), j, k;

    while (gt_table[i].key && gt_table[i].key != cmd)
        i = (i + 1) % GT_TABLE_SIZE;
    if (!gt_table[i].key)
        return;
    for (j = (i + 1) % GT_TABLE_SIZE; gt_table[j].key; j = (j + 1) % GT_TABLE_SIZE)
    {
        k = gt_hash(gt_table[j].key);
        if ((j > i && (k <= i || k > j)) || (j < i && (k <= i && k > j)))
        {
            gt_table[i] = gt_table[j];
            i = j;
        }
    }
    __atomic_store_n(&gt_table[i].key, NULL, __ATOMIC_RELEASE);
    gt_table[i].cmd = NULL;
    gt_tracked--;
}

static void gt_mark(struct gt_cmd *c, char kind)
{
    if (c->count >= GT_MAX_QUERIES)
        return;
    gt_orig.vkCmdWriteTimestamp2(c->cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, c->pool, c->count);
    c->cur.kind = kind;
    c->segs[c->count++] = c->cur;
    memset(&c->cur, 0, sizeof(c->cur));
    c->cur.width = c->rp_width;
    c->cur.height = c->rp_height;
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_BeginCommandBuffer(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *info)
{
    VkQueryPoolCreateInfo pool_info;
    struct gt_cmd *c;
    uint32_t slot;
    VkResult vr;

    if ((vr = gt_orig.vkBeginCommandBuffer(cmd, info)) < 0)
        return vr;
    if (!__atomic_load_n(&gt_window_frame, __ATOMIC_RELAXED) || (info->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT))
    {
        /* Begun again outside the window without having been submitted: stop tracking it. */
        if (gt_lookup(cmd))
        {
            pthread_mutex_lock(&gt_lock);
            if ((c = gt_lookup(cmd)))
            {
                gt_table_remove(cmd);
                gt_orig.vkResetQueryPool(gt_device, c->pool, 0, GT_MAX_QUERIES);
                c->next = gt_free;
                gt_free = c;
            }
            pthread_mutex_unlock(&gt_lock);
        }
        return vr;
    }

    pthread_mutex_lock(&gt_lock);
    if ((c = gt_lookup(cmd)))
        gt_orig.vkResetQueryPool(gt_device, c->pool, 0, GT_MAX_QUERIES);
    else
    {
        if ((c = gt_free))
            gt_free = c->next;
        else if (gt_allocated < GT_MAX_CMDS && (c = vkd3d_calloc(1, sizeof(*c))))
        {
            memset(&pool_info, 0, sizeof(pool_info));
            pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            pool_info.queryCount = GT_MAX_QUERIES;
            if (gt_orig.vkCreateQueryPool(gt_device, &pool_info, NULL, &c->pool) < 0)
            {
                vkd3d_free(c);
                c = NULL;
            }
            else
            {
                /* Host reset (hostQueryReset stays enabled for D3D11on12): works for command
                 * buffers of transfer-only queues too, where vkCmdResetQueryPool is not allowed. */
                gt_orig.vkResetQueryPool(gt_device, c->pool, 0, GT_MAX_QUERIES);
                gt_allocated++;
            }
        }
        if (c && gt_tracked < GT_TABLE_SIZE / 2)
        {
            for (slot = gt_hash(cmd); gt_table[slot].key; slot = (slot + 1) % GT_TABLE_SIZE)
                ;
            gt_table[slot].cmd = c;
            __atomic_store_n(&gt_table[slot].key, cmd, __ATOMIC_RELEASE);
            gt_tracked++;
        }
        else if (c)
        {
            c->next = gt_free;
            gt_free = c;
            c = NULL;
        }
    }
    pthread_mutex_unlock(&gt_lock);
    if (!c)
        return vr;

    c->cmd = cmd;
    c->frame = gt_window_frame;
    c->submitted = false;
    c->rp_width = c->rp_height = 0;
    memset(&c->cur, 0, sizeof(c->cur));
    c->count = 0;
    c->note_count = 0;
    gt_mark(c, 's');
    return vr;
}

static void gt_note_c(struct gt_cmd *c, const char *fmt, ...);

static void gt_note(struct gt_cmd *c, const char *fmt, va_list args)
{
    if (c->note_count >= GT_MAX_NOTES)
        return;
    c->notes[c->note_count].at = c->count;
    vsnprintf(c->notes[c->note_count].text, sizeof(c->notes[0].text), fmt, args);
    c->note_count++;
}

/* D3D12-level annotations for sampled command buffers ("GN frame queue seq segment text"). */
void vkd3d_gputime_note(VkCommandBuffer cmd, const char *fmt, ...)
{
    struct gt_cmd *c;
    va_list args;

    if (!(c = gt_lookup(cmd)))
        return;
    va_start(args, fmt);
    gt_note(c, fmt, args);
    va_end(args);
}

/* VKD3D_BENCH_GPUTIME_STACKS=1: the game's call stack (exe RVAs) for a note, to find the code that
 * records a pass. */
void vkd3d_gputime_note_stack(VkCommandBuffer cmd, const char *what)
{
    static int enabled = -1;
    char text[124], value[16];
    void *frames[16];
    uintptr_t exe;
    struct gt_cmd *c;
    unsigned int i, n;
    int len;

    if (!(c = gt_lookup(cmd)))
        return;
    if (enabled < 0)
        enabled = vkd3d_get_env_var("VKD3D_BENCH_GPUTIME_STACKS", value, sizeof(value)) && value[0] == '1';
    if (!enabled)
        return;
    exe = (uintptr_t)GetModuleHandleA(NULL);
    n = RtlCaptureStackBackTrace(1, ARRAY_SIZE(frames), frames, NULL);
    len = snprintf(text, sizeof(text), "stack %s:", what);
    for (i = 0; i < n && len < (int)sizeof(text) - 12; i++)
    {
        uintptr_t a = (uintptr_t)frames[i];
        if (a >= exe && a < exe + 0x9000000)
            len += snprintf(text + len, sizeof(text) - len, " %#x", (unsigned int)(a - exe));
    }
    gt_note_c(c, "%s", text);
}

bool vkd3d_gputime_tracking(VkCommandBuffer cmd)
{
    return !!gt_lookup(cmd);
}

static void gt_note_c(struct gt_cmd *c, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    gt_note(c, fmt, args);
    va_end(args);
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_EndCommandBuffer(VkCommandBuffer cmd)
{
    struct gt_cmd *c;

    if ((c = gt_lookup(cmd)))
        gt_mark(c, 'z');
    return gt_orig.vkEndCommandBuffer(cmd);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdPipelineBarrier2(VkCommandBuffer cmd, const VkDependencyInfo *info)
{
    struct gt_cmd *c;

    gt_orig.vkCmdPipelineBarrier2(cmd, info);
    if ((c = gt_lookup(cmd)))
    {
        VkPipelineStageFlags2 src = 0, dst = 0;
        VkAccessFlags2 srca = 0, dsta = 0;
        uint32_t i, layouts = 0;

        for (i = 0; i < info->memoryBarrierCount; i++)
        {
            src |= info->pMemoryBarriers[i].srcStageMask; dst |= info->pMemoryBarriers[i].dstStageMask;
            srca |= info->pMemoryBarriers[i].srcAccessMask; dsta |= info->pMemoryBarriers[i].dstAccessMask;
        }
        for (i = 0; i < info->bufferMemoryBarrierCount; i++)
        {
            src |= info->pBufferMemoryBarriers[i].srcStageMask; dst |= info->pBufferMemoryBarriers[i].dstStageMask;
            srca |= info->pBufferMemoryBarriers[i].srcAccessMask; dsta |= info->pBufferMemoryBarriers[i].dstAccessMask;
        }
        for (i = 0; i < info->imageMemoryBarrierCount; i++)
        {
            src |= info->pImageMemoryBarriers[i].srcStageMask; dst |= info->pImageMemoryBarriers[i].dstStageMask;
            srca |= info->pImageMemoryBarriers[i].srcAccessMask; dsta |= info->pImageMemoryBarriers[i].dstAccessMask;
            layouts += info->pImageMemoryBarriers[i].oldLayout != info->pImageMemoryBarriers[i].newLayout;
        }
        gt_note_c(c, "vk barrier mem %u buf %u img %u (layout changes %u) stages %#"PRIx64" -> %#"PRIx64" access %#"PRIx64" -> %#"PRIx64,
                info->memoryBarrierCount, info->bufferMemoryBarrierCount, info->imageMemoryBarrierCount, layouts,
                (uint64_t)src, (uint64_t)dst, (uint64_t)srca, (uint64_t)dsta);
        gt_mark(c, 'b');
    }
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdBeginRendering(VkCommandBuffer cmd, const VkRenderingInfo *info)
{
    struct gt_cmd *c;

    if ((c = gt_lookup(cmd)))
    {
        gt_mark(c, 'r');
        c->rp_width = c->cur.width = info->renderArea.extent.width;
        c->rp_height = c->cur.height = info->renderArea.extent.height;
    }
    gt_orig.vkCmdBeginRendering(cmd, info);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdEndRendering(VkCommandBuffer cmd)
{
    struct gt_cmd *c;

    gt_orig.vkCmdEndRendering(cmd);
    if ((c = gt_lookup(cmd)))
    {
        gt_mark(c, 'e');
        c->rp_width = c->rp_height = c->cur.width = c->cur.height = 0;
    }
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdBindPipeline(VkCommandBuffer cmd, VkPipelineBindPoint bind_point, VkPipeline pipeline)
{
    struct gt_cmd *c;

    gt_orig.vkCmdBindPipeline(cmd, bind_point, pipeline);
    if ((c = gt_lookup(cmd)))
    {
        if (!c->cur.binds++)
            c->cur.pipeline = (uint64_t)pipeline;
    }
}

#define GT_COUNT(field, amount) do { struct gt_cmd *c_; if ((c_ = gt_lookup(cmd))) c_->cur.field += (amount); } while (0)

static VKAPI_ATTR void VKAPI_CALL gt_CmdDraw(VkCommandBuffer cmd, uint32_t a, uint32_t b, uint32_t d, uint32_t e)
{
    GT_COUNT(draws, 1);
    gt_orig.vkCmdDraw(cmd, a, b, d, e);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDrawIndexed(VkCommandBuffer cmd, uint32_t a, uint32_t b, uint32_t d, int32_t e, uint32_t f)
{
    GT_COUNT(draws, 1);
    gt_orig.vkCmdDrawIndexed(cmd, a, b, d, e, f);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDrawIndirect(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, uint32_t count, uint32_t stride)
{
    GT_COUNT(draws, count);
    gt_orig.vkCmdDrawIndirect(cmd, buffer, offset, count, stride);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDrawIndexedIndirect(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, uint32_t count, uint32_t stride)
{
    GT_COUNT(draws, count);
    gt_orig.vkCmdDrawIndexedIndirect(cmd, buffer, offset, count, stride);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDrawIndirectCount(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset,
        VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_count, uint32_t stride)
{
    GT_COUNT(draws, 1);
    gt_orig.vkCmdDrawIndirectCount(cmd, buffer, offset, count_buffer, count_offset, max_count, stride);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDrawIndexedIndirectCount(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset,
        VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_count, uint32_t stride)
{
    GT_COUNT(draws, 1);
    gt_orig.vkCmdDrawIndexedIndirectCount(cmd, buffer, offset, count_buffer, count_offset, max_count, stride);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z)
{
    struct gt_cmd *c;

    if ((c = gt_lookup(cmd)))
    {
        c->cur.dispatches++;
        c->cur.groups += (uint64_t)x * y * z;
    }
    gt_orig.vkCmdDispatch(cmd, x, y, z);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdDispatchIndirect(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset)
{
    GT_COUNT(dispatches, 1);
    gt_orig.vkCmdDispatchIndirect(cmd, buffer, offset);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdExecuteGeneratedCommandsEXT(VkCommandBuffer cmd, VkBool32 preprocessed,
        const VkGeneratedCommandsInfoEXT *info)
{
    GT_COUNT(draws, 1);
    gt_orig.vkCmdExecuteGeneratedCommandsEXT(cmd, preprocessed, info);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdCopyBuffer2(VkCommandBuffer cmd, const VkCopyBufferInfo2 *info)
{
    GT_COUNT(copies, 1);
    gt_orig.vkCmdCopyBuffer2(cmd, info);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdCopyImage2(VkCommandBuffer cmd, const VkCopyImageInfo2 *info)
{
    GT_COUNT(copies, 1);
    gt_orig.vkCmdCopyImage2(cmd, info);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdCopyBufferToImage2(VkCommandBuffer cmd, const VkCopyBufferToImageInfo2 *info)
{
    GT_COUNT(copies, 1);
    gt_orig.vkCmdCopyBufferToImage2(cmd, info);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdCopyImageToBuffer2(VkCommandBuffer cmd, const VkCopyImageToBufferInfo2 *info)
{
    GT_COUNT(copies, 1);
    gt_orig.vkCmdCopyImageToBuffer2(cmd, info);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdFillBuffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data)
{
    GT_COUNT(clears, 1);
    gt_orig.vkCmdFillBuffer(cmd, buffer, offset, size, data);
}

static VKAPI_ATTR void VKAPI_CALL gt_CmdClearAttachments(VkCommandBuffer cmd, uint32_t count, const VkClearAttachment *attachments,
        uint32_t rect_count, const VkClearRect *rects)
{
    GT_COUNT(clears, 1);
    gt_orig.vkCmdClearAttachments(cmd, count, attachments, rect_count, rects);
}

/* CPU time on the same scale as NVIDIA's GPU timer (ns since the Unix epoch). */
static uint64_t gt_epoch_ns(void)
{
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    return ((((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime) - 116444736000000000ull) * 100;
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_QueueSubmit2(VkQueue queue, uint32_t count, const VkSubmitInfo2 *submits, VkFence fence)
{
    uint64_t now = gt_epoch_ns();
    struct gt_cmd *c;
    uint32_t i, j;

    if (__atomic_load_n(&gt_tracked, __ATOMIC_RELAXED))
    {
        pthread_mutex_lock(&gt_lock);
        for (i = 0; i < count; i++)
        {
            for (j = 0; j < submits[i].commandBufferInfoCount; j++)
            {
                if (!(c = gt_lookup(submits[i].pCommandBufferInfos[j].commandBuffer)))
                    continue;
                gt_table_remove(c->cmd);
                c->queue = queue;
                c->seq = ++gt_seq;
                c->submit_ns = now;
                c->submitted = true;
                c->next = gt_pending;
                gt_pending = c;
            }
        }
        pthread_mutex_unlock(&gt_lock);
    }
    return gt_orig.vkQueueSubmit2(queue, count, submits, fence);
}

void vkd3d_gputime_install(struct d3d12_device *device)
{
    struct vkd3d_vk_device_procs *vk = &device->vk_procs;
    char path[VKD3D_PATH_MAX], every[16];

    if (!vkd3d_get_env_var("VKD3D_BENCH_GPUTIME", path, sizeof(path)) || !path[0])
        return;
    /* Every device is hooked: the game creates and destroys a first device before the real one.
     * The latest device owns the query pools (a device that never presents never samples). */
    if (!gt_file)
    {
        if (!(gt_file = fopen(path, "w")))
        {
            ERR("bench gputime: failed to open %s.\n", path);
            return;
        }
        setvbuf(gt_file, NULL, _IOFBF, 1 << 16);
        if (vkd3d_get_env_var("VKD3D_BENCH_GPUTIME_EVERY", every, sizeof(every)) && atoi(every) > 0)
            gt_every = atoi(every);
        if (vkd3d_get_env_var("VKD3D_BENCH_GPUTIME_RUN", every, sizeof(every)) && atoi(every) > 0)
            gt_run = min(atoi(every), gt_every);
        gt_period = device->vk_info.device_limits.timestampPeriod;
        fprintf(gt_file, "# gputime v2 period %.3f every %u run %u\n", gt_period, gt_every, gt_run);
    }
    pthread_mutex_lock(&gt_lock);
    gt_device = device->vk_device;
    gt_orig = *vk;
    gt_vk = vk;
    pthread_mutex_unlock(&gt_lock);

    vk->vkBeginCommandBuffer = gt_BeginCommandBuffer;
    vk->vkEndCommandBuffer = gt_EndCommandBuffer;
    vk->vkCmdPipelineBarrier2 = gt_CmdPipelineBarrier2;
    vk->vkCmdBeginRendering = gt_CmdBeginRendering;
    vk->vkCmdEndRendering = gt_CmdEndRendering;
    vk->vkCmdBindPipeline = gt_CmdBindPipeline;
    vk->vkCmdDraw = gt_CmdDraw;
    vk->vkCmdDrawIndexed = gt_CmdDrawIndexed;
    vk->vkCmdDrawIndirect = gt_CmdDrawIndirect;
    vk->vkCmdDrawIndexedIndirect = gt_CmdDrawIndexedIndirect;
    if (vk->vkCmdDrawIndirectCount)
        vk->vkCmdDrawIndirectCount = gt_CmdDrawIndirectCount;
    if (vk->vkCmdDrawIndexedIndirectCount)
        vk->vkCmdDrawIndexedIndirectCount = gt_CmdDrawIndexedIndirectCount;
    vk->vkCmdDispatch = gt_CmdDispatch;
    vk->vkCmdDispatchIndirect = gt_CmdDispatchIndirect;
    if (vk->vkCmdExecuteGeneratedCommandsEXT)
        vk->vkCmdExecuteGeneratedCommandsEXT = gt_CmdExecuteGeneratedCommandsEXT;
    vk->vkCmdCopyBuffer2 = gt_CmdCopyBuffer2;
    vk->vkCmdCopyImage2 = gt_CmdCopyImage2;
    vk->vkCmdCopyBufferToImage2 = gt_CmdCopyBufferToImage2;
    vk->vkCmdCopyImageToBuffer2 = gt_CmdCopyImageToBuffer2;
    vk->vkCmdFillBuffer = gt_CmdFillBuffer;
    vk->vkCmdClearAttachments = gt_CmdClearAttachments;
    vk->vkQueueSubmit2 = gt_QueueSubmit2;
    INFO("bench gputime: sampling one frame in %u into %s.\n", gt_every, path);
}

/* From d3d12_command_list_Dispatch, before the vkCmdDispatch it labels. */
void vkd3d_gputime_dispatch_label(VkCommandBuffer cmd, uint64_t shader_hash, uint32_t x, uint32_t y, uint32_t z)
{
    struct gt_cmd *c;

    if (!(c = gt_lookup(cmd)) || c->cur.shader)
        return;
    c->cur.shader = shader_hash;
    c->cur.dims[0] = x;
    c->cur.dims[1] = y;
    c->cur.dims[2] = z;
}

/* Called at every present: open/close the sampling window, write out finished command buffers. */
void vkd3d_gputime_present(void)
{
    uint64_t ts[GT_MAX_QUERIES], prev;
    struct gt_cmd **p, *c;
    const struct gt_seg *s;
    uint32_t i;
    VkResult vr;

    if (!gt_file)
        return;

    /* GP frame cpu_epoch_ns: the present that ends frame <frame>'s recording, for the frame before
     * the sampled ones (it opens the window) and the sampled ones. */
    if (gt_window_frame || (gt_presents + 1) % gt_every == 0)
        fprintf(gt_file, "GP\t%"PRIu64"\t%"PRIu64"\n", gt_presents, gt_epoch_ns());
    gt_presents++;
    __atomic_store_n(&gt_window_frame, gt_presents % gt_every < gt_run ? gt_presents : 0, __ATOMIC_RELAXED);

    pthread_mutex_lock(&gt_lock);
    for (p = &gt_pending; (c = *p);)
    {
        vr = c->count ? gt_orig.vkGetQueryPoolResults(gt_device, c->pool, 0, c->count, sizeof(ts), ts,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) : VK_SUCCESS;
        if (vr == VK_NOT_READY && gt_presents - c->frame < 600)
        {
            p = &c->next;
            continue;
        }
        *p = c->next;
        if (vr == VK_SUCCESS && c->count)
        {
            fprintf(gt_file, "GS\t%"PRIu64"\t%p\t%"PRIu64"\t%"PRIu64"\t%.3f\n", c->frame, (void *)c->queue, c->seq,
                    (uint64_t)(ts[0] * gt_period), c->submit_ns / 1e6);
            prev = ts[0];
            for (i = 1; i < c->count; i++)
            {
                s = &c->segs[i];
                fprintf(gt_file, "GT\t%"PRIu64"\t%p\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%c\t%u\t%u\t%"PRIu64"\t%u\t%u\t%u\t%ux%u\t%"PRIx64"\t%016"PRIx64"\t%ux%ux%u\n",
                        c->frame, (void *)c->queue, c->seq, (uint64_t)(ts[i] * gt_period),
                        (uint64_t)((ts[i] - prev) * gt_period), s->kind, s->draws, s->dispatches, s->groups,
                        s->copies, s->clears, s->binds, s->width, s->height, s->pipeline, s->shader,
                        s->dims[0], s->dims[1], s->dims[2]);
                prev = ts[i];
            }
            for (i = 0; i < c->note_count; i++)
                fprintf(gt_file, "GN\t%"PRIu64"\t%p\t%"PRIu64"\t%u\t%s\n", c->frame, (void *)c->queue, c->seq,
                        c->notes[i].at, c->notes[i].text);
        }
        gt_orig.vkResetQueryPool(gt_device, c->pool, 0, GT_MAX_QUERIES);
        c->next = gt_free;
        gt_free = c;
    }
    pthread_mutex_unlock(&gt_lock);
    fflush(gt_file);
}
