/*
 * Deferred command recording (miles-bench, VKD3D_BENCH_CMD_REPLAY=1).
 *
 * Miles Morales records nearly all of its rendering on one thread ("Render Thread"), and that thread
 * is the CPU critical path; ~20% of it is the Vulkan driver (vkCmd* through winevulkan into
 * libnvidia-glcore) recording what vkd3d already worked out. Here the command buffers of D3D12
 * command lists record into a stream instead: each vkCmd* call appends its arguments (deep-copied)
 * to the command buffer's stream, and one worker thread replays the streams into the real command
 * buffers, in order, while the render thread goes on. The driver sees the same calls with the same
 * arguments in the same order on each command buffer, so the GPU work is identical.
 *
 * Handles: command buffers allocated from a command allocator's pools (registered in command.c) are
 * handed to vkd3d as a tagged pointer (bit 0 set) to their struct rp_cb. Every device function that
 * can see a command buffer is wrapped (cmdreplay_gen.py generates the wrappers and fails the build
 * script if one is missing): a tagged handle is recorded or untagged, never passed to the driver.
 *
 * Ordering and Vulkan's external synchronization:
 *   - A command buffer's ops sit in its own stream (chunks), so a list handed between threads keeps
 *     its order. A chunk is published to the worker's FIFO when it is full enough, and at
 *     vkEndCommandBuffer; the worker replays chunks in publication order.
 *   - A command pool and its buffers are touched only by the worker while it has their ops, so pool
 *     operations (allocate, free, reset, destroy) first wait until the worker has finished
 *     everything published (and drop the unpublished ops of buffers they reset or free: those
 *     commands are discarded by the operation anyway).
 *   - Submission waits until the worker has replayed each command buffer's vkEndCommandBuffer.
 *   - vkCmd* calls without a recorder (rare ones) publish the buffer's stream, wait for the worker,
 *     and call the driver directly on the recording thread (the worker is then done with the pool).
 *   - Destroying or freeing an object waits for every published op first: a replayed op may name it.
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

struct rp_op
{
    void (*replay)(VkCommandBuffer cb, const struct rp_op *op);
    uint32_t size;
};

struct rp_chunk
{
    struct rp_chunk *next;
    VkCommandBuffer vk_cb;
    uint64_t seq;
    uint32_t used, cap, ops, pad;
    uint64_t pad2; /* data[] 16-byte aligned */
    uint8_t data[];
};

struct rp_pool
{
    VkCommandPool vk_pool;
    struct rp_cb *cbs;
    struct rp_pool *next;
};

struct rp_cb
{
    VkCommandBuffer vk_cb;
    struct rp_pool *pool;
    struct rp_cb *pool_next;
    struct rp_chunk *chunk; /* ops not yet published */
    uint64_t end_seq;       /* chunk holding the last vkEndCommandBuffer */
};

#define RP_TAGGED(cb) ((uintptr_t)(cb) & 1)
#define RP_CB(cb) ((struct rp_cb *)((uintptr_t)(cb) & ~(uintptr_t)1))
#define RP_HANDLE(c) ((VkCommandBuffer)((uintptr_t)(c) | 1))

#define RP_CHUNK_SIZE (64u << 10)
#define RP_FREE_MAX 64 /* chunks kept for reuse */
/* A chunk is published once it holds this many ops (at the next op): the worker replays a long
 * command buffer while it is still being recorded. */
#define RP_PUBLISH_OPS 96
/* Pause loops before sleeping: the worker between chunks, waiters for the worker. */
#define RP_WORKER_SPIN 8000
#define RP_WAIT_SPIN 4000

static struct vkd3d_vk_device_procs rp_real;

static struct
{
    bool enabled;
    pthread_mutex_t lock;
    pthread_cond_t work_cond, done_cond;
    struct rp_chunk *head, *tail, *free;
    unsigned int free_count;
    uint64_t published; /* written under lock */
    uint64_t done;      /* last chunk replayed */
    LONG waiters;
    bool sleeping;
    pthread_t thread;

    pthread_mutex_t pool_lock;
    struct rp_pool *pools;

    /* Measurement: ops recorded, chunks replayed, direct calls by name, waits. */
    uint64_t ops, chunks, guard_waits, submit_waits, pool_waits, wait_us, dropped, deferred_destroys;
    uint64_t drains[];
} *rp_state;

#define rp (*rp_state)

static void rp_spin_pause(void)
{
#if defined(__i386__) || defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

static uint64_t rp_now_us(void)
{
    LARGE_INTEGER t, f;

    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (uint64_t)(t.QuadPart * 1000000.0 / f.QuadPart);
}

static struct rp_chunk *rp_chunk_get(VkCommandBuffer vk_cb, uint32_t min_size, bool exact)
{
    struct rp_chunk *k = NULL;
    uint32_t cap = exact ? min_size : max(RP_CHUNK_SIZE, min_size);

    if (cap == RP_CHUNK_SIZE)
    {
        pthread_mutex_lock(&rp.lock);
        if ((k = rp.free))
        {
            rp.free = k->next;
            rp.free_count--;
        }
        pthread_mutex_unlock(&rp.lock);
    }
    if (!k && !(k = vkd3d_malloc(sizeof(*k) + cap)))
        return NULL;
    k->next = NULL;
    k->vk_cb = vk_cb;
    k->seq = 0;
    k->used = 0;
    k->cap = cap;
    k->ops = 0;
    return k;
}

static void rp_chunk_put(struct rp_chunk *k)
{
    if (k->cap != RP_CHUNK_SIZE)
    {
        vkd3d_free(k);
        return;
    }
    pthread_mutex_lock(&rp.lock);
    if (rp.free_count < RP_FREE_MAX)
    {
        k->next = rp.free;
        rp.free = k;
        rp.free_count++;
        k = NULL;
    }
    pthread_mutex_unlock(&rp.lock);
    vkd3d_free(k);
}

static uint64_t rp_publish_chunk(struct rp_chunk *k)
{
    uint64_t seq;
    bool wake;

    pthread_mutex_lock(&rp.lock);
    seq = k->seq = rp.published + 1;
    __atomic_store_n(&rp.published, seq, __ATOMIC_RELEASE);
    if (rp.tail)
        rp.tail->next = k;
    else
        __atomic_store_n(&rp.head, k, __ATOMIC_RELEASE);
    rp.tail = k;
    wake = rp.sleeping;
    pthread_mutex_unlock(&rp.lock);
    if (wake)
        pthread_cond_signal(&rp.work_cond);
    return seq;
}

/* Publishes the buffer's pending ops; returns the sequence number the worker reaches once they are
 * replayed (the last published one when there was nothing to publish). */
static uint64_t rp_publish(struct rp_cb *c)
{
    struct rp_chunk *k = c->chunk;

    c->chunk = NULL;
    if (k && !k->ops)
    {
        rp_chunk_put(k);
        k = NULL;
    }
    if (!k)
        return __atomic_load_n(&rp.published, __ATOMIC_ACQUIRE);
    return rp_publish_chunk(k);
}

/* A destroy queued behind everything published (no command buffer): a chunk of its own holding one op,
 * or NULL when the worker has nothing left to replay (the caller destroys right away). */
static struct rp_chunk *rp_defer_chunk(void (*replay)(VkCommandBuffer, const struct rp_op *), size_t size)
{
    struct rp_chunk *k;
    struct rp_op *op;

    if (__atomic_load_n(&rp.done, __ATOMIC_ACQUIRE) >= __atomic_load_n(&rp.published, __ATOMIC_ACQUIRE))
        return NULL;
    size = align(size, 16);
    if (!(k = rp_chunk_get(VK_NULL_HANDLE, size, true)))
        return NULL;
    op = (struct rp_op *)k->data;
    op->replay = replay;
    op->size = size;
    k->used = size;
    k->ops = 1;
    __atomic_add_fetch(&rp.deferred_destroys, 1, __ATOMIC_RELAXED);
    return k;
}

static void rp_drop(struct rp_cb *c)
{
    if (c->chunk)
    {
        rp_chunk_put(c->chunk);
        c->chunk = NULL;
        __atomic_add_fetch(&rp.dropped, 1, __ATOMIC_RELAXED);
    }
}

static void rp_wait(uint64_t seq, uint64_t *counter)
{
    uint64_t start;
    unsigned int i;

    if (__atomic_load_n(&rp.done, __ATOMIC_ACQUIRE) >= seq)
        return;
    start = rp_now_us();
    __atomic_add_fetch(counter, 1, __ATOMIC_RELAXED);
    for (i = 0; i < RP_WAIT_SPIN; i++)
    {
        if (__atomic_load_n(&rp.done, __ATOMIC_ACQUIRE) >= seq)
            goto done;
        rp_spin_pause();
    }
    pthread_mutex_lock(&rp.lock);
    __atomic_add_fetch(&rp.waiters, 1, __ATOMIC_SEQ_CST);
    while (__atomic_load_n(&rp.done, __ATOMIC_SEQ_CST) < seq)
        pthread_cond_wait(&rp.done_cond, &rp.lock);
    __atomic_sub_fetch(&rp.waiters, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&rp.lock);
done:
    __atomic_add_fetch(&rp.wait_us, rp_now_us() - start, __ATOMIC_RELAXED);
}

/* Everything published so far has been replayed. */
static void rp_guard(void)
{
    rp_wait(__atomic_load_n(&rp.published, __ATOMIC_ACQUIRE), &rp.guard_waits);
}

static void *rp_op_alloc(struct rp_cb *c, void (*replay)(VkCommandBuffer, const struct rp_op *), size_t size)
{
    struct rp_chunk *k = c->chunk;
    struct rp_op *op;

    size = align(size, 16);
    if (k && (k->ops >= RP_PUBLISH_OPS || k->used + size > k->cap))
    {
        rp_publish(c);
        k = NULL;
    }
    if (!k && !(k = c->chunk = rp_chunk_get(c->vk_cb, size, false)))
    {
        ERR("cmdreplay: out of memory.\n");
        abort();
    }
    op = (struct rp_op *)(k->data + k->used);
    k->used += size;
    k->ops++;
    op->replay = replay;
    op->size = size;
    return op;
}

/* For a call made directly: the worker has replayed everything this buffer published before. */
static VkCommandBuffer rp_drain_for_direct(struct rp_cb *c, unsigned int index)
{
    __atomic_add_fetch(&rp.drains[index], 1, __ATOMIC_RELAXED);
    rp_wait(rp_publish(c), &rp.pool_waits);
    return c->vk_cb;
}

#include "cmdreplay_gen.h"

/* Hand-written recorders that fell back to a direct call (an extension struct they do not copy),
 * counted after the generated drains. */
enum rp_hand
{
    RP_HAND_BEGIN = RP_DRAIN_COUNT,
    RP_HAND_BARRIER,
    RP_HAND_RENDERING,
    RP_HAND_BIND_DB,
    RP_HAND_COPY,
    RP_HAND_PUSH_DESC,
    RP_HAND_EXEC,
    RP_HAND_COUNT,
};

static const char *const rp_hand_names[] =
{
    "vkBeginCommandBuffer", "vkCmdPipelineBarrier2", "vkCmdBeginRendering", "vkCmdBindDescriptorBuffersEXT",
    "vkCmdCopy*2", "vkCmdPushDescriptorSetKHR", "vkCmdExecuteCommands",
};

static void rp_worker_replay(struct rp_chunk *k)
{
    const uint8_t *p = k->data, *end = k->data + k->used;
    const struct rp_op *op;

    while (p < end)
    {
        op = (const struct rp_op *)p;
        op->replay(k->vk_cb, op);
        p += op->size;
    }
}

static void *rp_worker_main(void *arg)
{
    struct rp_chunk *k;
    unsigned int i;

    (void)arg;
    vkd3d_set_thread_name("vkd3d_cmdrep");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    for (;;)
    {
        pthread_mutex_lock(&rp.lock);
        while (!rp.head)
        {
            pthread_mutex_unlock(&rp.lock);
            for (i = 0; i < RP_WORKER_SPIN && !__atomic_load_n(&rp.head, __ATOMIC_ACQUIRE); i++)
                rp_spin_pause();
            pthread_mutex_lock(&rp.lock);
            if (rp.head)
                break;
            rp.sleeping = true;
            pthread_cond_wait(&rp.work_cond, &rp.lock);
            rp.sleeping = false;
        }
        k = rp.head;
        __atomic_store_n(&rp.head, k->next, __ATOMIC_RELEASE);
        if (!rp.head)
            rp.tail = NULL;
        pthread_mutex_unlock(&rp.lock);

        rp_worker_replay(k);
        rp.ops += k->ops;
        rp.chunks++;
        __atomic_store_n(&rp.done, k->seq, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&rp.waiters, __ATOMIC_SEQ_CST))
        {
            pthread_mutex_lock(&rp.lock);
            pthread_cond_broadcast(&rp.done_cond);
            pthread_mutex_unlock(&rp.lock);
        }
        rp_chunk_put(k);
    }
    return NULL;
}

/* Copies count elements of size bytes after the op; NULL stays NULL. */
#define RP_ARRAY(dst, src, count) do { \
        if ((src) && (count)) { memcpy(payload, (src), sizeof(*(src)) * (count)); (dst) = (void *)payload; \
            payload += align(sizeof(*(src)) * (count), 16); } else (dst) = NULL; } while (0)
#define RP_ARRAY_SIZE(src, count) (((src) && (count)) ? align(sizeof(*(src)) * (count), 16) : 0)

/* Command buffer lifetime */

struct rp_op_begin
{
    struct rp_op h;
    VkCommandBufferUsageFlags flags;
};

static void rp_replay_begin(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_begin *op = (const void *)h;
    VkCommandBufferBeginInfo info;
    VkResult vr;

    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = NULL;
    info.flags = op->flags;
    info.pInheritanceInfo = NULL;
    if ((vr = rp_real.vkBeginCommandBuffer(cb, &info)) < 0)
        ERR("cmdreplay: vkBeginCommandBuffer failed, vr %d.\n", vr);
}

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkBeginCommandBuffer(VkCommandBuffer commandBuffer,
        const VkCommandBufferBeginInfo *info)
{
    struct rp_op_begin *op;
    struct rp_cb *c;

    if (!RP_TAGGED(commandBuffer))
        return rp_real.vkBeginCommandBuffer(commandBuffer, info);
    c = RP_CB(commandBuffer);
    if (info->pNext || info->pInheritanceInfo)
        return rp_real.vkBeginCommandBuffer(rp_drain_for_direct(c, RP_HAND_BEGIN), info);
    op = rp_op_alloc(c, rp_replay_begin, sizeof(*op));
    op->flags = info->flags;
    return VK_SUCCESS;
}

static void rp_replay_end(VkCommandBuffer cb, const struct rp_op *h)
{
    VkResult vr;

    (void)h;
    if ((vr = rp_real.vkEndCommandBuffer(cb)) < 0)
        ERR("cmdreplay: vkEndCommandBuffer failed, vr %d.\n", vr);
}

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkEndCommandBuffer(VkCommandBuffer commandBuffer)
{
    struct rp_cb *c;

    if (!RP_TAGGED(commandBuffer))
        return rp_real.vkEndCommandBuffer(commandBuffer);
    c = RP_CB(commandBuffer);
    rp_op_alloc(c, rp_replay_end, sizeof(struct rp_op));
    c->end_seq = rp_publish(c);
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkResetCommandBuffer(VkCommandBuffer commandBuffer,
        VkCommandBufferResetFlags flags)
{
    struct rp_cb *c;

    if (!RP_TAGGED(commandBuffer))
        return rp_real.vkResetCommandBuffer(commandBuffer, flags);
    c = RP_CB(commandBuffer);
    rp_drop(c);
    rp_wait(__atomic_load_n(&rp.published, __ATOMIC_ACQUIRE), &rp.pool_waits);
    return rp_real.vkResetCommandBuffer(c->vk_cb, flags);
}

/* Pools */

static struct rp_pool *rp_pool_find(VkCommandPool vk_pool)
{
    struct rp_pool *pool;

    pthread_mutex_lock(&rp.pool_lock);
    for (pool = rp.pools; pool; pool = pool->next)
        if (pool->vk_pool == vk_pool)
            break;
    pthread_mutex_unlock(&rp.pool_lock);
    return pool;
}

void vkd3d_cmd_replay_register_pool(VkCommandPool vk_pool)
{
    struct rp_pool *pool;

    if (!rp_state || !vk_pool || rp_pool_find(vk_pool))
        return;
    if (!(pool = vkd3d_calloc(1, sizeof(*pool))))
        return;
    pool->vk_pool = vk_pool;
    pthread_mutex_lock(&rp.pool_lock);
    pool->next = rp.pools;
    rp.pools = pool;
    pthread_mutex_unlock(&rp.pool_lock);
}

static void rp_pool_unlink(struct rp_pool *pool, struct rp_cb *c);

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkAllocateCommandBuffers(VkDevice device,
        const VkCommandBufferAllocateInfo *info, VkCommandBuffer *cbs)
{
    struct rp_pool *pool;
    struct rp_cb *c;
    uint32_t i, j;
    VkResult vr;

    if (!(pool = rp_pool_find(info->commandPool)) || info->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
    {
        if (pool)
            rp_guard();
        return rp_real.vkAllocateCommandBuffers(device, info, cbs);
    }
    rp_guard();
    if ((vr = rp_real.vkAllocateCommandBuffers(device, info, cbs)) < 0)
        return vr;
    for (i = 0; i < info->commandBufferCount; i++)
    {
        if (!(c = vkd3d_calloc(1, sizeof(*c))))
        {
            for (j = 0; j < i; j++)
            {
                c = RP_CB(cbs[j]);
                rp_pool_unlink(pool, c);
                cbs[j] = c->vk_cb;
                vkd3d_free(c);
            }
            rp_real.vkFreeCommandBuffers(device, info->commandPool, info->commandBufferCount, cbs);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        c->vk_cb = cbs[i];
        c->pool = pool;
        c->pool_next = pool->cbs;
        pool->cbs = c;
        cbs[i] = RP_HANDLE(c);
    }
    return VK_SUCCESS;
}

static void rp_pool_unlink(struct rp_pool *pool, struct rp_cb *c)
{
    struct rp_cb **p;

    for (p = &pool->cbs; *p; p = &(*p)->pool_next)
    {
        if (*p == c)
        {
            *p = c->pool_next;
            return;
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL rp_vkFreeCommandBuffers(VkDevice device, VkCommandPool vk_pool,
        uint32_t count, const VkCommandBuffer *cbs)
{
    VkCommandBuffer stack[16], *real = stack;
    struct rp_cb *c;
    uint32_t i;

    rp_guard();
    if (count > ARRAY_SIZE(stack) && !(real = vkd3d_malloc(count * sizeof(*real))))
        return;
    for (i = 0; i < count; i++)
    {
        if (!RP_TAGGED(cbs[i]))
        {
            real[i] = cbs[i];
            continue;
        }
        c = RP_CB(cbs[i]);
        rp_drop(c);
        rp_pool_unlink(c->pool, c);
        real[i] = c->vk_cb;
        vkd3d_free(c);
    }
    rp_real.vkFreeCommandBuffers(device, vk_pool, count, real);
    if (real != stack)
        vkd3d_free(real);
}

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkResetCommandPool(VkDevice device, VkCommandPool vk_pool,
        VkCommandPoolResetFlags flags)
{
    struct rp_pool *pool;
    struct rp_cb *c;

    if ((pool = rp_pool_find(vk_pool)))
        for (c = pool->cbs; c; c = c->pool_next)
            rp_drop(c);
    /* Unregistered pools too: a replayed vkCmdExecuteCommands may name their secondaries. */
    rp_guard();
    return rp_real.vkResetCommandPool(device, vk_pool, flags);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkDestroyCommandPool(VkDevice device, VkCommandPool vk_pool,
        const VkAllocationCallbacks *allocator)
{
    struct rp_pool *pool, **p;
    struct rp_cb *c, *next;

    pthread_mutex_lock(&rp.pool_lock);
    for (p = &rp.pools; (pool = *p); p = &pool->next)
    {
        if (pool->vk_pool == vk_pool)
        {
            *p = pool->next;
            break;
        }
    }
    pthread_mutex_unlock(&rp.pool_lock);
    if (pool)
    {
        for (c = pool->cbs; c; c = c->pool_next)
            rp_drop(c);
    }
    rp_guard();
    rp_real.vkDestroyCommandPool(device, vk_pool, allocator);
    if (pool)
    {
        for (c = pool->cbs; c; c = next)
        {
            next = c->pool_next;
            vkd3d_free(c);
        }
        vkd3d_free(pool);
    }
}

static VKAPI_ATTR void VKAPI_CALL rp_vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *allocator)
{
    /* Queued destroys run before the device goes. */
    rp_guard();
    rp_real.vkDestroyDevice(device, allocator);
}

/* Submission */

static uint64_t rp_untag(VkCommandBuffer *cb)
{
    struct rp_cb *c;

    if (!RP_TAGGED(*cb))
        return 0;
    c = RP_CB(*cb);
    *cb = c->vk_cb;
    return c->end_seq;
}

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkQueueSubmit2(VkQueue queue, uint32_t count,
        const VkSubmitInfo2 *submits, VkFence fence)
{
    VkCommandBufferSubmitInfo stack_cbs[64], *cbs = stack_cbs;
    VkSubmitInfo2 stack_submits[16], *copy = stack_submits;
    uint32_t i, j, total = 0, n = 0;
    uint64_t seq, wait = 0;
    bool tagged = false;
    VkResult vr;

    for (i = 0; i < count; i++)
    {
        total += submits[i].commandBufferInfoCount;
        for (j = 0; j < submits[i].commandBufferInfoCount; j++)
            tagged |= RP_TAGGED(submits[i].pCommandBufferInfos[j].commandBuffer);
    }
    if (!tagged)
        return rp_real.vkQueueSubmit2(queue, count, submits, fence);

    if (total > ARRAY_SIZE(stack_cbs) && !(cbs = vkd3d_malloc(total * sizeof(*cbs))))
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (count > ARRAY_SIZE(stack_submits) && !(copy = vkd3d_malloc(count * sizeof(*copy))))
    {
        if (cbs != stack_cbs)
            vkd3d_free(cbs);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (i = 0; i < count; i++)
    {
        copy[i] = submits[i];
        copy[i].pCommandBufferInfos = &cbs[n];
        for (j = 0; j < submits[i].commandBufferInfoCount; j++, n++)
        {
            cbs[n] = submits[i].pCommandBufferInfos[j];
            if ((seq = rp_untag(&cbs[n].commandBuffer)) > wait)
                wait = seq;
        }
    }
    rp_wait(wait, &rp.submit_waits);
    vr = rp_real.vkQueueSubmit2(queue, count, copy, fence);
    if (cbs != stack_cbs)
        vkd3d_free(cbs);
    if (copy != stack_submits)
        vkd3d_free(copy);
    return vr;
}

/* Debug names may name a command buffer. */

static VKAPI_ATTR VkResult VKAPI_CALL rp_vkSetDebugUtilsObjectNameEXT(VkDevice device,
        const VkDebugUtilsObjectNameInfoEXT *info)
{
    VkDebugUtilsObjectNameInfoEXT copy = *info;

    if (copy.objectType == VK_OBJECT_TYPE_COMMAND_BUFFER && (copy.objectHandle & 1))
        copy.objectHandle = (uint64_t)(uintptr_t)RP_CB((VkCommandBuffer)(uintptr_t)copy.objectHandle)->vk_cb;
    return rp_real.vkSetDebugUtilsObjectNameEXT(device, &copy);
}

/* Recorders for commands that take pointers: the pointed-to data is copied after the op. A pNext
 * these do not know falls back to a direct call. */

struct rp_op_barrier
{
    struct rp_op h;
    VkDependencyInfo info;
};

static void rp_replay_barrier(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_barrier *op = (const void *)h;
    rp_real.vkCmdPipelineBarrier2(cb, &op->info);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdPipelineBarrier2(VkCommandBuffer commandBuffer,
        const VkDependencyInfo *info)
{
    struct rp_op_barrier *op;
    uint8_t *payload;
    struct rp_cb *c;
    uint32_t i;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdPipelineBarrier2(commandBuffer, info);
        return;
    }
    c = RP_CB(commandBuffer);
    if (info->pNext)
        goto direct;
    for (i = 0; i < info->memoryBarrierCount; i++)
        if (info->pMemoryBarriers[i].pNext)
            goto direct;
    for (i = 0; i < info->bufferMemoryBarrierCount; i++)
        if (info->pBufferMemoryBarriers[i].pNext)
            goto direct;
    for (i = 0; i < info->imageMemoryBarrierCount; i++)
        if (info->pImageMemoryBarriers[i].pNext)
            goto direct;
    op = rp_op_alloc(c, rp_replay_barrier, sizeof(*op) + 16
            + RP_ARRAY_SIZE(info->pMemoryBarriers, info->memoryBarrierCount)
            + RP_ARRAY_SIZE(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount)
            + RP_ARRAY_SIZE(info->pImageMemoryBarriers, info->imageMemoryBarrierCount));
    payload = (uint8_t *)(op + 1);
    payload += (uintptr_t)-(uintptr_t)payload & 15;
    op->info = *info;
    RP_ARRAY(op->info.pMemoryBarriers, info->pMemoryBarriers, info->memoryBarrierCount);
    RP_ARRAY(op->info.pBufferMemoryBarriers, info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
    RP_ARRAY(op->info.pImageMemoryBarriers, info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
    return;

direct:
    rp_real.vkCmdPipelineBarrier2(rp_drain_for_direct(c, RP_HAND_BARRIER), info);
}

struct rp_op_rendering
{
    struct rp_op h;
    VkRenderingInfo info;
    VkRenderingAttachmentInfo depth, stencil;
};

static void rp_replay_rendering(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_rendering *op = (const void *)h;
    rp_real.vkCmdBeginRendering(cb, &op->info);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdBeginRendering(VkCommandBuffer commandBuffer,
        const VkRenderingInfo *info)
{
    struct rp_op_rendering *op;
    uint8_t *payload;
    struct rp_cb *c;
    uint32_t i;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdBeginRendering(commandBuffer, info);
        return;
    }
    c = RP_CB(commandBuffer);
    if (info->pNext || (info->pDepthAttachment && info->pDepthAttachment->pNext) ||
            (info->pStencilAttachment && info->pStencilAttachment->pNext))
        goto direct;
    for (i = 0; i < info->colorAttachmentCount; i++)
        if (info->pColorAttachments[i].pNext)
            goto direct;
    op = rp_op_alloc(c, rp_replay_rendering, sizeof(*op) + 16
            + RP_ARRAY_SIZE(info->pColorAttachments, info->colorAttachmentCount));
    payload = (uint8_t *)(op + 1);
    payload += (uintptr_t)-(uintptr_t)payload & 15;
    op->info = *info;
    RP_ARRAY(op->info.pColorAttachments, info->pColorAttachments, info->colorAttachmentCount);
    if (info->pDepthAttachment)
    {
        op->depth = *info->pDepthAttachment;
        op->info.pDepthAttachment = &op->depth;
    }
    if (info->pStencilAttachment)
    {
        op->stencil = *info->pStencilAttachment;
        op->info.pStencilAttachment = &op->stencil;
    }
    return;

direct:
    rp_real.vkCmdBeginRendering(rp_drain_for_direct(c, RP_HAND_RENDERING), info);
}

struct rp_op_push_constants
{
    struct rp_op h;
    VkPipelineLayout layout;
    VkShaderStageFlags stages;
    uint32_t offset, size;
    uint8_t data[];
};

static void rp_replay_push_constants(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_push_constants *op = (const void *)h;
    rp_real.vkCmdPushConstants(cb, op->layout, op->stages, op->offset, op->size, op->data);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
        VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void *values)
{
    struct rp_op_push_constants *op;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdPushConstants(commandBuffer, layout, stages, offset, size, values);
        return;
    }
    op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_push_constants, sizeof(*op) + size);
    op->layout = layout;
    op->stages = stages;
    op->offset = offset;
    op->size = size;
    memcpy(op->data, values, size);
}

struct rp_op_viewports
{
    struct rp_op h;
    uint32_t first, count;
    bool with_count;
    VkViewport viewports[];
};

static void rp_replay_viewports(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_viewports *op = (const void *)h;
    if (op->with_count)
        rp_real.vkCmdSetViewportWithCount(cb, op->count, op->viewports);
    else
        rp_real.vkCmdSetViewport(cb, op->first, op->count, op->viewports);
}

static void rp_record_viewports(struct rp_cb *c, bool with_count, uint32_t first, uint32_t count,
        const VkViewport *viewports)
{
    struct rp_op_viewports *op;

    op = rp_op_alloc(c, rp_replay_viewports, sizeof(*op) + count * sizeof(*viewports));
    op->first = first;
    op->count = count;
    op->with_count = with_count;
    memcpy(op->viewports, viewports, count * sizeof(*viewports));
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdSetViewport(VkCommandBuffer commandBuffer, uint32_t first,
        uint32_t count, const VkViewport *viewports)
{
    if (!RP_TAGGED(commandBuffer))
        rp_real.vkCmdSetViewport(commandBuffer, first, count, viewports);
    else
        rp_record_viewports(RP_CB(commandBuffer), false, first, count, viewports);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdSetViewportWithCount(VkCommandBuffer commandBuffer, uint32_t count,
        const VkViewport *viewports)
{
    if (!RP_TAGGED(commandBuffer))
        rp_real.vkCmdSetViewportWithCount(commandBuffer, count, viewports);
    else
        rp_record_viewports(RP_CB(commandBuffer), true, 0, count, viewports);
}

struct rp_op_scissors
{
    struct rp_op h;
    uint32_t first, count;
    bool with_count;
    VkRect2D rects[];
};

static void rp_replay_scissors(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_scissors *op = (const void *)h;
    if (op->with_count)
        rp_real.vkCmdSetScissorWithCount(cb, op->count, op->rects);
    else
        rp_real.vkCmdSetScissor(cb, op->first, op->count, op->rects);
}

static void rp_record_scissors(struct rp_cb *c, bool with_count, uint32_t first, uint32_t count,
        const VkRect2D *rects)
{
    struct rp_op_scissors *op;

    op = rp_op_alloc(c, rp_replay_scissors, sizeof(*op) + count * sizeof(*rects));
    op->first = first;
    op->count = count;
    op->with_count = with_count;
    memcpy(op->rects, rects, count * sizeof(*rects));
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdSetScissor(VkCommandBuffer commandBuffer, uint32_t first,
        uint32_t count, const VkRect2D *rects)
{
    if (!RP_TAGGED(commandBuffer))
        rp_real.vkCmdSetScissor(commandBuffer, first, count, rects);
    else
        rp_record_scissors(RP_CB(commandBuffer), false, first, count, rects);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdSetScissorWithCount(VkCommandBuffer commandBuffer, uint32_t count,
        const VkRect2D *rects)
{
    if (!RP_TAGGED(commandBuffer))
        rp_real.vkCmdSetScissorWithCount(commandBuffer, count, rects);
    else
        rp_record_scissors(RP_CB(commandBuffer), true, 0, count, rects);
}

struct rp_op_blend_constants
{
    struct rp_op h;
    float constants[4];
};

static void rp_replay_blend_constants(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_blend_constants *op = (const void *)h;
    rp_real.vkCmdSetBlendConstants(cb, op->constants);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdSetBlendConstants(VkCommandBuffer commandBuffer, const float constants[4])
{
    struct rp_op_blend_constants *op;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdSetBlendConstants(commandBuffer, constants);
        return;
    }
    op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_blend_constants, sizeof(*op));
    memcpy(op->constants, constants, sizeof(op->constants));
}

struct rp_op_db_offsets
{
    struct rp_op h;
    VkPipelineBindPoint bind_point;
    VkPipelineLayout layout;
    uint32_t first_set, set_count;
    uint32_t *indices;
    VkDeviceSize offsets[];
};

static void rp_replay_db_offsets(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_db_offsets *op = (const void *)h;
    rp_real.vkCmdSetDescriptorBufferOffsetsEXT(cb, op->bind_point, op->layout, op->first_set, op->set_count,
            op->indices, op->offsets);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdSetDescriptorBufferOffsetsEXT(VkCommandBuffer commandBuffer,
        VkPipelineBindPoint bind_point, VkPipelineLayout layout, uint32_t first_set, uint32_t set_count,
        const uint32_t *indices, const VkDeviceSize *offsets)
{
    struct rp_op_db_offsets *op;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdSetDescriptorBufferOffsetsEXT(commandBuffer, bind_point, layout, first_set, set_count,
                indices, offsets);
        return;
    }
    op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_db_offsets,
            sizeof(*op) + set_count * (sizeof(*offsets) + sizeof(*indices)));
    op->bind_point = bind_point;
    op->layout = layout;
    op->first_set = first_set;
    op->set_count = set_count;
    memcpy(op->offsets, offsets, set_count * sizeof(*offsets));
    op->indices = (uint32_t *)(op->offsets + set_count);
    memcpy(op->indices, indices, set_count * sizeof(*indices));
}

struct rp_op_bind_db
{
    struct rp_op h;
    uint32_t count;
    VkDescriptorBufferBindingPushDescriptorBufferHandleEXT handles[2];
    VkDescriptorBufferBindingInfoEXT infos[];
};

static void rp_replay_bind_db(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_bind_db *op = (const void *)h;
    rp_real.vkCmdBindDescriptorBuffersEXT(cb, op->count, op->infos);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdBindDescriptorBuffersEXT(VkCommandBuffer commandBuffer, uint32_t count,
        const VkDescriptorBufferBindingInfoEXT *infos)
{
    const VkDescriptorBufferBindingPushDescriptorBufferHandleEXT *handle;
    struct rp_op_bind_db *op;
    unsigned int handles = 0;
    struct rp_cb *c;
    uint32_t i;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdBindDescriptorBuffersEXT(commandBuffer, count, infos);
        return;
    }
    c = RP_CB(commandBuffer);
    for (i = 0; i < count; i++)
    {
        if (!(handle = infos[i].pNext))
            continue;
        if (handle->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE_EXT ||
                handle->pNext || ++handles > ARRAY_SIZE(op->handles))
        {
            rp_real.vkCmdBindDescriptorBuffersEXT(rp_drain_for_direct(c, RP_HAND_BIND_DB), count, infos);
            return;
        }
    }
    op = rp_op_alloc(c, rp_replay_bind_db, sizeof(*op) + count * sizeof(*infos));
    op->count = count;
    memcpy(op->infos, infos, count * sizeof(*infos));
    for (i = 0, handles = 0; i < count; i++)
    {
        if (!infos[i].pNext)
            continue;
        op->handles[handles] = *(const VkDescriptorBufferBindingPushDescriptorBufferHandleEXT *)infos[i].pNext;
        op->infos[i].pNext = &op->handles[handles++];
    }
}

struct rp_op_vertex_buffers
{
    struct rp_op h;
    uint32_t first, count;
    const VkDeviceSize *sizes, *strides;
    VkDeviceSize *offsets;
    VkBuffer buffers[];
};

static void rp_replay_vertex_buffers(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_vertex_buffers *op = (const void *)h;
    rp_real.vkCmdBindVertexBuffers2(cb, op->first, op->count, op->buffers, op->offsets, op->sizes, op->strides);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t first,
        uint32_t count, const VkBuffer *buffers, const VkDeviceSize *offsets, const VkDeviceSize *sizes,
        const VkDeviceSize *strides)
{
    struct rp_op_vertex_buffers *op;
    VkDeviceSize *p;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdBindVertexBuffers2(commandBuffer, first, count, buffers, offsets, sizes, strides);
        return;
    }
    op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_vertex_buffers,
            sizeof(*op) + count * (sizeof(*buffers) + 3 * sizeof(VkDeviceSize)));
    op->first = first;
    op->count = count;
    memcpy(op->buffers, buffers, count * sizeof(*buffers));
    p = (VkDeviceSize *)(op->buffers + count);
    op->offsets = p;
    memcpy(p, offsets, count * sizeof(*p));
    p += count;
    op->sizes = sizes ? memcpy(p, sizes, count * sizeof(*p)) : NULL;
    p += count;
    op->strides = strides ? memcpy(p, strides, count * sizeof(*p)) : NULL;
}

struct rp_op_descriptor_sets
{
    struct rp_op h;
    VkPipelineBindPoint bind_point;
    VkPipelineLayout layout;
    uint32_t first_set, set_count, dynamic_count;
    const uint32_t *dynamic;
    VkDescriptorSet sets[];
};

static void rp_replay_descriptor_sets(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_descriptor_sets *op = (const void *)h;
    rp_real.vkCmdBindDescriptorSets(cb, op->bind_point, op->layout, op->first_set, op->set_count, op->sets,
            op->dynamic_count, op->dynamic);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer,
        VkPipelineBindPoint bind_point, VkPipelineLayout layout, uint32_t first_set, uint32_t set_count,
        const VkDescriptorSet *sets, uint32_t dynamic_count, const uint32_t *dynamic)
{
    struct rp_op_descriptor_sets *op;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdBindDescriptorSets(commandBuffer, bind_point, layout, first_set, set_count, sets,
                dynamic_count, dynamic);
        return;
    }
    op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_descriptor_sets,
            sizeof(*op) + set_count * sizeof(*sets) + dynamic_count * sizeof(*dynamic));
    op->bind_point = bind_point;
    op->layout = layout;
    op->first_set = first_set;
    op->set_count = set_count;
    op->dynamic_count = dynamic_count;
    memcpy(op->sets, sets, set_count * sizeof(*sets));
    op->dynamic = dynamic_count ? memcpy(op->sets + set_count, dynamic, dynamic_count * sizeof(*dynamic)) : NULL;
}

struct rp_op_update_buffer
{
    struct rp_op h;
    VkBuffer buffer;
    VkDeviceSize offset, size;
    uint8_t data[];
};

static void rp_replay_update_buffer(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_update_buffer *op = (const void *)h;
    rp_real.vkCmdUpdateBuffer(cb, op->buffer, op->offset, op->size, op->data);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
        VkDeviceSize offset, VkDeviceSize size, const void *data)
{
    struct rp_op_update_buffer *op;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdUpdateBuffer(commandBuffer, buffer, offset, size, data);
        return;
    }
    op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_update_buffer, sizeof(*op) + size);
    op->buffer = buffer;
    op->offset = offset;
    op->size = size;
    memcpy(op->data, data, size);
}

/* Copies: the info struct and its regions. */
struct rp_op_copy
{
    struct rp_op h;
    union
    {
        VkCopyBufferInfo2 buffer;
        VkCopyBufferToImageInfo2 to_image;
        VkCopyImageToBufferInfo2 to_buffer;
        VkCopyImageInfo2 image;
    } u;
};

static void rp_replay_copy_buffer(VkCommandBuffer cb, const struct rp_op *h)
{
    rp_real.vkCmdCopyBuffer2(cb, &((const struct rp_op_copy *)(const void *)h)->u.buffer);
}

static void rp_replay_copy_to_image(VkCommandBuffer cb, const struct rp_op *h)
{
    rp_real.vkCmdCopyBufferToImage2(cb, &((const struct rp_op_copy *)(const void *)h)->u.to_image);
}

static void rp_replay_copy_to_buffer(VkCommandBuffer cb, const struct rp_op *h)
{
    rp_real.vkCmdCopyImageToBuffer2(cb, &((const struct rp_op_copy *)(const void *)h)->u.to_buffer);
}

static void rp_replay_copy_image(VkCommandBuffer cb, const struct rp_op *h)
{
    rp_real.vkCmdCopyImage2(cb, &((const struct rp_op_copy *)(const void *)h)->u.image);
}

#define RP_COPY_RECORDER(fn, info_type, member, replay) \
static VKAPI_ATTR void VKAPI_CALL rp_##fn(VkCommandBuffer commandBuffer, const info_type *info) \
{ \
    struct rp_op_copy *op; \
    struct rp_cb *c; \
    uint32_t i; \
\
    if (!RP_TAGGED(commandBuffer)) \
    { \
        rp_real.fn(commandBuffer, info); \
        return; \
    } \
    c = RP_CB(commandBuffer); \
    if (info->pNext) \
        goto direct; \
    for (i = 0; i < info->regionCount; i++) \
        if (info->pRegions[i].pNext) \
            goto direct; \
    op = rp_op_alloc(c, replay, sizeof(*op) + info->regionCount * sizeof(*info->pRegions)); \
    op->u.member = *info; \
    op->u.member.pRegions = memcpy(op + 1, info->pRegions, info->regionCount * sizeof(*info->pRegions)); \
    return; \
direct: \
    rp_real.fn(rp_drain_for_direct(c, RP_HAND_COPY), info); \
}

RP_COPY_RECORDER(vkCmdCopyBuffer2, VkCopyBufferInfo2, buffer, rp_replay_copy_buffer)
RP_COPY_RECORDER(vkCmdCopyBufferToImage2, VkCopyBufferToImageInfo2, to_image, rp_replay_copy_to_image)
RP_COPY_RECORDER(vkCmdCopyImageToBuffer2, VkCopyImageToBufferInfo2, to_buffer, rp_replay_copy_to_buffer)
RP_COPY_RECORDER(vkCmdCopyImage2, VkCopyImageInfo2, image, rp_replay_copy_image)

struct rp_op_push_descriptors
{
    struct rp_op h;
    VkPipelineBindPoint bind_point;
    VkPipelineLayout layout;
    uint32_t set, count;
    VkWriteDescriptorSet writes[];
};

static void rp_replay_push_descriptors(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_push_descriptors *op = (const void *)h;
    rp_real.vkCmdPushDescriptorSetKHR(cb, op->bind_point, op->layout, op->set, op->count, op->writes);
}

/* The array a write's descriptorType reads; the others may be left pointing anywhere. */
static const VkDescriptorImageInfo *rp_write_images(const VkWriteDescriptorSet *w)
{
    switch (w->descriptorType)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            return w->pImageInfo;
        default:
            return NULL;
    }
}

static const VkDescriptorBufferInfo *rp_write_buffers(const VkWriteDescriptorSet *w)
{
    switch (w->descriptorType)
    {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            return w->pBufferInfo;
        default:
            return NULL;
    }
}

static const VkBufferView *rp_write_texels(const VkWriteDescriptorSet *w)
{
    switch (w->descriptorType)
    {
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            return w->pTexelBufferView;
        default:
            return NULL;
    }
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer,
        VkPipelineBindPoint bind_point, VkPipelineLayout layout, uint32_t set, uint32_t count,
        const VkWriteDescriptorSet *writes)
{
    struct rp_op_push_descriptors *op;
    size_t size = 0;
    uint8_t *payload;
    struct rp_cb *c;
    uint32_t i;

    if (!RP_TAGGED(commandBuffer))
    {
        rp_real.vkCmdPushDescriptorSetKHR(commandBuffer, bind_point, layout, set, count, writes);
        return;
    }
    c = RP_CB(commandBuffer);
    for (i = 0; i < count; i++)
    {
        if (writes[i].pNext)
        {
            rp_real.vkCmdPushDescriptorSetKHR(rp_drain_for_direct(c, RP_HAND_PUSH_DESC), bind_point, layout, set,
                    count, writes);
            return;
        }
        size += RP_ARRAY_SIZE(rp_write_images(&writes[i]), writes[i].descriptorCount)
                + RP_ARRAY_SIZE(rp_write_buffers(&writes[i]), writes[i].descriptorCount)
                + RP_ARRAY_SIZE(rp_write_texels(&writes[i]), writes[i].descriptorCount);
    }
    op = rp_op_alloc(c, rp_replay_push_descriptors, sizeof(*op) + count * sizeof(*writes) + 16 + size);
    op->bind_point = bind_point;
    op->layout = layout;
    op->set = set;
    op->count = count;
    memcpy(op->writes, writes, count * sizeof(*writes));
    payload = (uint8_t *)(op->writes + count);
    payload += (uintptr_t)-(uintptr_t)payload & 15;
    for (i = 0; i < count; i++)
    {
        RP_ARRAY(op->writes[i].pImageInfo, rp_write_images(&writes[i]), writes[i].descriptorCount);
        RP_ARRAY(op->writes[i].pBufferInfo, rp_write_buffers(&writes[i]), writes[i].descriptorCount);
        RP_ARRAY(op->writes[i].pTexelBufferView, rp_write_texels(&writes[i]), writes[i].descriptorCount);
    }
}

struct rp_op_execute
{
    struct rp_op h;
    uint32_t count;
    VkCommandBuffer cbs[];
};

static void rp_replay_execute(VkCommandBuffer cb, const struct rp_op *h)
{
    const struct rp_op_execute *op = (const void *)h;
    rp_real.vkCmdExecuteCommands(cb, op->count, op->cbs);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t count,
        const VkCommandBuffer *cbs)
{
    VkCommandBuffer stack[16], *real = stack;
    struct rp_op_execute *op;
    uint32_t i;

    if (RP_TAGGED(commandBuffer))
    {
        op = rp_op_alloc(RP_CB(commandBuffer), rp_replay_execute, sizeof(*op) + count * sizeof(*cbs));
        op->count = count;
        for (i = 0; i < count; i++)
            op->cbs[i] = RP_TAGGED(cbs[i]) ? RP_CB(cbs[i])->vk_cb : cbs[i];
        return;
    }
    if (count > ARRAY_SIZE(stack) && !(real = vkd3d_malloc(count * sizeof(*real))))
        return;
    for (i = 0; i < count; i++)
        real[i] = RP_TAGGED(cbs[i]) ? RP_CB(cbs[i])->vk_cb : cbs[i];
    rp_real.vkCmdExecuteCommands(commandBuffer, count, real);
    if (real != stack)
        vkd3d_free(real);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdPreprocessGeneratedCommandsEXT(VkCommandBuffer commandBuffer,
        const VkGeneratedCommandsInfoEXT *info, VkCommandBuffer stateCommandBuffer)
{
    if (RP_TAGGED(commandBuffer))
        commandBuffer = rp_drain_for_direct(RP_CB(commandBuffer), RP_HAND_EXEC);
    if (RP_TAGGED(stateCommandBuffer))
        stateCommandBuffer = rp_drain_for_direct(RP_CB(stateCommandBuffer), RP_HAND_EXEC);
    rp_real.vkCmdPreprocessGeneratedCommandsEXT(commandBuffer, info, stateCommandBuffer);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdPreprocessGeneratedCommandsNV(VkCommandBuffer commandBuffer,
        const VkGeneratedCommandsInfoNV *info)
{
    if (RP_TAGGED(commandBuffer))
        commandBuffer = rp_drain_for_direct(RP_CB(commandBuffer), RP_HAND_EXEC);
    rp_real.vkCmdPreprocessGeneratedCommandsNV(commandBuffer, info);
}

static VKAPI_ATTR void VKAPI_CALL rp_vkCmdExecuteGeneratedCommandsNV(VkCommandBuffer commandBuffer,
        VkBool32 isPreprocessed, const VkGeneratedCommandsInfoNV *info)
{
    if (RP_TAGGED(commandBuffer))
        commandBuffer = rp_drain_for_direct(RP_CB(commandBuffer), RP_HAND_EXEC);
    rp_real.vkCmdExecuteGeneratedCommandsNV(commandBuffer, isPreprocessed, info);
}

/* Stats, from vkd3d_cmd_replay_report() (each present). */
void vkd3d_cmd_replay_report(void)
{
    static uint64_t last_us, last_ops, last_chunks;
    char text[1024];
    uint64_t now, ops, chunks;
    size_t len;
    unsigned int i;

    if (!rp_state || (now = rp_now_us()) - last_us < 20000000)
        return;
    if (!last_us)
    {
        last_us = now;
        return;
    }
    ops = __atomic_load_n(&rp.ops, __ATOMIC_RELAXED);
    chunks = __atomic_load_n(&rp.chunks, __ATOMIC_RELAXED);
    len = snprintf(text, sizeof(text), "cmdreplay: %.0f ops/s in %.0f chunks/s; waits: submit %"PRIu64", pool %"PRIu64
            ", guard %"PRIu64", %.1f ms total; %"PRIu64" destroys queued, %"PRIu64" chunks dropped; direct:",
            (ops - last_ops) * 1e6 / (now - last_us), (chunks - last_chunks) * 1e6 / (now - last_us),
            rp.submit_waits, rp.pool_waits, rp.guard_waits, rp.wait_us / 1000.0, rp.deferred_destroys, rp.dropped);
    for (i = 0; i < RP_HAND_COUNT && len < sizeof(text) - 64; i++)
    {
        if (rp.drains[i])
            len += snprintf(text + len, sizeof(text) - len, " %s %"PRIu64,
                    i < RP_DRAIN_COUNT ? rp_drain_names[i] : rp_hand_names[i - RP_DRAIN_COUNT], rp.drains[i]);
    }
    INFO("%s.\n", text);
    last_us = now;
    last_ops = ops;
    last_chunks = chunks;
}

#define RP_GEN_INSTALL
#include "cmdreplay_gen.h"

void vkd3d_cmd_replay_install(struct d3d12_device *device)
{
    char value[8];

    if (!vkd3d_get_env_var("VKD3D_BENCH_CMD_REPLAY", value, sizeof(value)) || value[0] != '1')
        return;
    if (!rp_state)
    {
        if (!(rp_state = vkd3d_calloc(1, sizeof(*rp_state) + RP_HAND_COUNT * sizeof(rp_state->drains[0]))))
            return;
        rp_real = device->vk_procs;
        pthread_mutex_init(&rp.lock, NULL);
        pthread_mutex_init(&rp.pool_lock, NULL);
        pthread_cond_init(&rp.work_cond, NULL);
        pthread_cond_init(&rp.done_cond, NULL);
        if (pthread_create(&rp.thread, NULL, rp_worker_main, NULL))
        {
            ERR("cmdreplay: failed to start the worker thread.\n");
            vkd3d_free(rp_state);
            rp_state = NULL;
            return;
        }
        INFO("cmdreplay: command lists record through the replay worker.\n");
    }
    else
    {
        /* Another device (games create a throwaway one first): fill what the first one lacked. The
         * winevulkan entry points are the same functions for every device. */
        const PFN_vkVoidFunction *src = (const PFN_vkVoidFunction *)&device->vk_procs;
        PFN_vkVoidFunction *dst = (PFN_vkVoidFunction *)&rp_real;
        size_t i;

        for (i = 0; i < sizeof(rp_real) / sizeof(*dst); i++)
        {
            if (!dst[i])
                dst[i] = src[i];
            else if (src[i] && src[i] != dst[i])
                WARN("cmdreplay: device entry point %zu differs from the first device's.\n", i);
        }
    }
    rp_install_procs(&device->vk_procs);
}

/* For code outside vkd3d that records into a list's command buffer itself (interop): the worker
 * replays everything recorded so far, then the caller gets the real handle. */
VkCommandBuffer vkd3d_cmd_replay_direct_handle(VkCommandBuffer cb)
{
    return RP_TAGGED(cb) ? rp_drain_for_direct(RP_CB(cb), RP_HAND_EXEC) : cb;
}

bool vkd3d_cmd_replay_enabled(void)
{
    return !!rp_state;
}
