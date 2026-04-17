#include "models.h"
#include <cmath>

ggml_cgraph * clip_graph_hunyuanocr::build() {
    const int merge = hparams.n_merge;
    const int pw    = n_patches_x;
    const int ph    = n_patches_y;

    // Position embedding interpolation.
    // HunyuanVL uses explicit scale factors (target+0.1)/n_grid to match Python's behavior.
    // HunyuanOCR uses the same square layout and the standard ratio-based interpolation.
    ggml_tensor * pos_embd = nullptr;
    if (proj_type == PROJECTOR_TYPE_HUNYUANVL && model.position_embeddings) {
        const int64_t n_pos  = model.position_embeddings->ne[1]; // n_grid * n_grid
        const int     n_grid = (int)std::round(std::sqrt((double)n_pos));
        ggml_tensor * pos_patch = model.position_embeddings;
        if (ph == n_grid && pw == n_grid) {
            pos_embd = pos_patch; // no interpolation needed
        } else {
            pos_patch = ggml_reshape_3d(ctx0, pos_patch, n_embd, n_grid, n_grid);
            pos_patch = ggml_permute(ctx0, pos_patch, 2, 0, 1, 3);
            pos_patch = ggml_cont(ctx0, pos_patch);
            pos_patch = ggml_interpolate_sf(ctx0, pos_patch, pw, ph, n_embd, 1,
                                            GGML_SCALE_MODE_BILINEAR,
                                            (float)(pw + 0.1f) / n_grid,
                                            (float)(ph + 0.1f) / n_grid);
            pos_patch = ggml_permute(ctx0, pos_patch, 1, 2, 0, 3);
            pos_embd  = ggml_cont_2d(ctx0, pos_patch, n_embd, ph * pw);
        }
    } else {
        pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR);
    }

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, pos_embd, nullptr);

    // perceiver projector
    cur = build_norm(cur, model.mm_pre_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);

    // [C, W*H] -> [W, H, C] for conv2d
    cur = ggml_reshape_3d(ctx0, cur, n_embd, pw, ph);
    cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
    cur = ggml_cont(ctx0, cur);

    // Conv2d(1152->2304, k=2, s=2) + GELU + Conv2d(2304->4608, k=1, s=1)
    cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, merge, merge, 0, 0, 1, 1);
    if (model.mm_0_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_0_b, 1, 1, model.mm_0_b->ne[0]));
    }
    cur = ggml_gelu(ctx0, cur);
    cur = ggml_conv_2d(ctx0, model.mm_1_w, cur, 1, 1, 0, 0, 1, 1);
    if (model.mm_1_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_1_b, 1, 1, model.mm_1_b->ne[0]));
    }

    const int ow   = pw / merge;
    const int oh   = ph / merge;
    const int idim = (int)cur->ne[2]; // OC = 4608

    // append newline along W (dim 0)
    ggml_tensor * nl = ggml_reshape_4d(ctx0, model.image_newline, 1, 1, idim, 1);
    nl = ggml_repeat_4d(ctx0, nl, 1, oh, idim, 1);
    cur = ggml_concat(ctx0, cur, nl, 0);

    // [OW+1, OH, OC] -> [OC, (OW+1)*OH]
    cur = ggml_permute(ctx0, cur, 1, 2, 0, 3);
    cur = ggml_cont_2d(ctx0, cur, idim, (ow + 1) * oh);

    // project to LLM hidden size
    cur = build_mm(model.mm_model_proj, cur);
    if (model.mm_model_proj_b) {
        cur = ggml_add(ctx0, cur, model.mm_model_proj_b);
    }

    // wrap with begin/end tokens
    cur = ggml_concat(ctx0, ggml_reshape_2d(ctx0, model.mm_img_begin, model.mm_img_begin->ne[0], 1), cur, 1);
    cur = ggml_concat(ctx0, cur, ggml_reshape_2d(ctx0, model.mm_img_end, model.mm_img_end->ne[0], 1), 1);

    cur = build_norm(cur, model.mm_post_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
