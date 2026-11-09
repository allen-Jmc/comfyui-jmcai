# -*- coding: utf-8 -*-
"""
JMCAI H3 参考条件加噪（Cond Noise Aug）

H3 把参考图 / Add Guide 的引导帧 / 参考视频都当作"干净条件行"注入（视觉条件时间步默认 0.999）。
条件越干净，模型越倾向于照搬它的统计量：接缝处与段首容易发油光、饱和度上冲、细节被抹平。

本节点往 conditioning 里写两个键，模型侧直接读取：

    comfy/ldm/minimax/model.py
        _cond_video_rows / _cond_audio_rows:
            r = aug * r + (1 - aug) * noise      # 按比例给条件 latent 混高斯噪声
        _forward:
            seg_t["cond"] = max(t_v, aug)        # 条件行的时间步标签也随之下降

作用对象是全部视觉条件行（ref_image_0 / ref_image_1 / 参考视频 / 引导帧），音频条件单独由 audio_aug 控制。

    visual_aug = 0.999  官方默认，完全不混噪
    visual_aug = 0.90~0.97  轻微混噪：油光 / 过饱和通常明显减轻，参考锁定力基本不变
    visual_aug < 0.85   参考开始"不牢"，人物长相与配色会漂移

移植自 ComfyUI-YCNodes-MiniMax-H3 的 h3_cond_noise_aug.py，按 JMCAI 节点规范
重命名为 JMCAI_H3CondNoiseAug，功能与参数保持一致。
"""

import node_helpers


class JMCAI_H3CondNoiseAug:
    """给 H3 的参考条件混噪，压制接缝发油光与过饱和。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "H3 正向条件（MiniMax H3 Reference to Video / Add Guide 的输出）。"}),
                "visual_aug": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "视觉条件混噪系数：1.0 = 不混噪（官方默认 0.999）；0.90~0.97 减轻油光/过饱和；低于 0.85 人物与配色会漂移。"}),
                "audio_aug": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "音频条件混噪系数：1.0 = 不动（官方默认）。"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "JMCAI❤/MiniMax H3/conditioning"
    DESCRIPTION = ("把 minimax_visual_cond_noise_aug / minimax_audio_cond_noise_aug 写进 conditioning，"
                   "给参考图与引导帧的 latent 按比例混噪，减轻接缝与段首的发油光、过饱和。")

    def apply(self, conditioning, visual_aug, audio_aug):
        return (node_helpers.conditioning_set_values(conditioning, {
            "minimax_visual_cond_noise_aug": float(visual_aug),
            "minimax_audio_cond_noise_aug": float(audio_aug),
        }),)


NODE_CLASS_MAPPINGS = {
    "JMCAI_H3CondNoiseAug": JMCAI_H3CondNoiseAug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_H3CondNoiseAug": "JMCAI❤ H3 参考条件加噪",
}
