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
        const x = "tmp_" + Math.floor(Math.random() * 1000);
        p[x] = { inputs: { images: [String(t.id), 0] }, class_type: "PreviewImage" };
        await api.fetchApi("/prompt", { method: "POST", body: JSON.stringify({ prompt: p }) });
    } catch (e) { }
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
                let tm = null;
                const q = () => {
                    if (tm) clearTimeout(tm);
                    tm = setTimeout(() => {
                        const v = this.widgets.find(x => x.name === "仅预览")?.value;
                        if (v) doRun(this); else if (app.queuePrompt) app.queuePrompt(0);
                    }, 500);
                };
                const ks = ["裁切范围", "涂鸦透明度", "涂鸦颜色", "目标缩放", "目标尺寸", "对齐倍数", "回贴边界微调"];
                this.widgets.forEach(w => {
                    if (ks.includes(w.name)) {
                        const b = w.callback;
                        w.callback = function () {
                            const res = b ? b.apply(this, arguments) : undefined;
                            q();
                            return res;
                        };
                    }
                });
                this.addWidget("button", "手动加载预览", null, () => {
                    const v = this.widgets.find(x => x.name === "仅预览")?.value;
                    if (v) doRun(this); else if (app.queuePrompt) app.queuePrompt(0);
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
    }
});
