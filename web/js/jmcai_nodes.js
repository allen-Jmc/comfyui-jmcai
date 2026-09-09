import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const getDeps = (g, id, s = new Set()) => {
    s.add(String(id));
    const n = g[String(id)];
    if (!n) return s;
    Object.values(n.inputs || {}).forEach(l => {
        if (Array.isArray(l) && l.length) {
            const o = String(l[0]);
            if (!s.has(o)) getDeps(g, o, s);
        }
    });
    return s;
};

async function doRun(t) {
    try {
        const { output } = await app.graphToPrompt();
        const d = getDeps(output, t.id);
        const p = {};
        d.forEach(i => { p[i] = output[i]; });

        // 生成一个唯一的临时 ID 供预览节点使用
        const x = "tmp_preview_" + t.id;
        p[x] = {
            inputs: { images: [String(t.id), 0] },
            class_type: "PreviewImage",
            _meta: { title: "临时预览" }
        };

        // 执行请求
        await api.fetchApi("/prompt", {
            method: "POST",
            body: JSON.stringify({
                prompt: p,
                front: true // 优先执行
            })
        });
    } catch (e) {
        console.error("[JMCAI] Preview failed:", e);
    }
}

app.registerExtension({
    name: "JMCAI.Nodes.ImageBatchMulti",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "JMCAI_ImageBatch_Multi") {
            const oc = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = oc ? oc.apply(this, arguments) : undefined;
                const w = this.widgets.find(x => x.name === "输入图像数量");
                if (!w) return r;
                const up = () => {
                    const c = w.value;
                    const cur = (this.inputs || []).filter(i => i.name.startsWith("image_")).length;
                    if (c > cur) for (let i = cur + 1; i <= c; i++) this.addInput(`image_${i}`, "IMAGE");
                    else if (c < cur) for (let i = cur; i > c; i--) {
                        const idx = this.inputs.findIndex(p => p.name === `image_${i}`);
                        if (idx !== -1) this.removeInput(idx);
                    }
                    if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
                };
                this.addWidget("button", "Update inputs", null, up);
                setTimeout(up, 20);
                const old = w.callback;
                w.callback = function () {
                    const res = old ? old.apply(this, arguments) : undefined;
                    up();
                    return res;
                };
                return r;
            };
        }
        if (nodeData.name === "JMCAI_LocalCropPreprocess") {
            const oc = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = oc ? oc.apply(this, arguments) : undefined;
                const ks = ["裁切范围", "涂鸦透明度", "涂鸦颜色", "目标宽度", "目标高度", "强制正方形", "对齐倍数", "回贴边界微调"];
                this.widgets.forEach(w => {
                    if (ks.includes(w.name)) {
                        const b = w.callback;
                        w.callback = function () {
                            const res = b ? b.apply(this, arguments) : undefined;
                            // 移除自动触发 Queue，遵循用户手动点击按钮的需求
                            return res;
                        };
                    }
                });
                this.addWidget("button", "加载预览图", null, () => {
                    doRun(this);
                });
                return r;
            };
            const ox = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (m) {
                if (ox) ox.apply(this, arguments);
                if (m.images) {
                    this.imgs = m.images.map(i => {
                        const g = new Image();
                        g.src = `/view?filename=${i.filename}&type=${i.type}&subfolder=${i.subfolder}&t=${Date.now()}`;
                        return g;
                    });
                    this.setDirtyCanvas(true);
                }
            };
        }

        // 自动清理 API 系列生图节点的旧英文残留端口并无缝平移连线
        const API_IMAGE_NODE_NAMES = [
            "JMCAI_OpenAIImageGenerate",
            "JMCAI_GeminiImageGenerate",
            "JMCAI_ModelArkImageGenerate",
            "JMCAI_QwenImageGenerate",
            "JMCAI_GLMImageGenerate",
            "JMCAI_GrokImageGenerate",
            "JMCAI_APIImageSave"
        ];

        if (API_IMAGE_NODE_NAMES.includes(nodeData.name)) {
            const oc = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = oc ? oc.apply(this, arguments) : undefined;
                setTimeout(() => {
                    cleanupLegacyInputs(this);
                }, 20);
                return r;
            };

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated ? onCreated.apply(this, arguments) : undefined;
                setTimeout(() => {
                    cleanupLegacyInputs(this);
                }, 30);
                return r;
            };

            // 增加右键菜单项方便用户随时一键清理旧端口
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }
                options.push({
                    content: "JMCAI: 清理重复旧英文端口",
                    callback: () => {
                        cleanupLegacyInputs(this);
                    }
                });
            };
        }
    },
    loadedGraphNode(node) {
        const API_IMAGE_NODE_NAMES = [
            "JMCAI_OpenAIImageGenerate",
            "JMCAI_GeminiImageGenerate",
            "JMCAI_ModelArkImageGenerate",
            "JMCAI_QwenImageGenerate",
            "JMCAI_GLMImageGenerate",
            "JMCAI_GrokImageGenerate",
            "JMCAI_APIImageSave"
        ];
        if (node && API_IMAGE_NODE_NAMES.includes(node.type || node.comfyClass)) {
            setTimeout(() => {
                cleanupLegacyInputs(node);
            }, 50);
        }
    }
});

const LEGACY_INPUT_MAP = {
    "ref_images": "多参考图批次",
    "image1": "参考图1",
    "image2": "参考图2",
    "image3": "参考图3",
    "mask": "局部重绘遮罩",
    "images": "图像"
};

function cleanupLegacyInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;

    let modified = false;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const slot = node.inputs[i];
        if (!slot) continue;
        const targetChineseName = LEGACY_INPUT_MAP[slot.name];
        if (!targetChineseName) continue;

        // 检查该节点中是否已经存在对应的中文新端口
        const targetIdx = node.inputs.findIndex(inp => inp && inp.name === targetChineseName);
        if (targetIdx !== -1 && targetIdx !== i) {
            // 如果旧英文端口上有连线
            if (slot.link != null) {
                const targetSlot = node.inputs[targetIdx];
                // 如果目标中文端口尚未连线，则将连线平滑迁移过去
                if (targetSlot && targetSlot.link == null) {
                    const linkId = slot.link;
                    const graph = node.graph || app.graph;
                    const linkObj = graph?.links?.[linkId] || graph?._links?.[linkId];
                    if (linkObj) {
                        linkObj.target_slot = targetIdx;
                        targetSlot.link = linkId;
                        slot.link = null;
                    }
                } else {
                    // 若中文端口已有连线，安全断开旧端口
                    if (typeof node.disconnectInput === "function") {
                        node.disconnectInput(i);
                    }
                }
            }
            // 从节点上移除残留的旧英文端口
            node.removeInput(i);
            modified = true;
        }
    }

    if (modified) {
        if (typeof node.computeSize === "function") {
            const sz = node.computeSize();
            node.size[0] = Math.max(node.size[0], sz[0]);
            node.size[1] = Math.max(node.size[1], sz[1]);
        }
        if (typeof node.setDirtyCanvas === "function") {
            node.setDirtyCanvas(true, true);
        }
    }
}
