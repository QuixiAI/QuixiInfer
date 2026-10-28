// Offline golden-file generator. Not part of the QuixiInfer build: it links llama.cpp, which runs only here,
// once, to record reference outputs that QuixiInfer's tests compare against.
//
// Build (see make.sh): links ~/llama.cpp/build-qx. Run:
//   golden MODEL.gguf OUT_DIR PROMPT_FILE... [-- options]
// Options (environment):
//   QX_GOLDEN_LOGITS_FROM=N   record logits only from position N on (default 0)
//   QX_GOLDEN_ACT_POS=P       record activations for positions < P (default 4)
//   QX_GOLDEN_ACT=a,b,c       activation name prefixes to record (default: a fixed list)
//   QX_GOLDEN_THREADS=T       CPU threads (default 16)
//   QX_GOLDEN_NGL=N           layers on the GPU (default 0: CPU only)
//   QX_GOLDEN_FA=1            keep flash attention and the F16 KV cache (default: off, F32 KV)
//   QX_GOLDEN_TOKENIZE_ONLY=1 write pK.tokens and stop (no model weights are read)
//
// Each prompt file is UTF-8 text, tokenized without special tokens and decoded one token at a time, so every
// position takes the single-token decode path. Outputs per prompt k:
//   pK.tokens   int32 token ids
//   pK.logits   float32 [n_recorded][n_vocab], positions from QX_GOLDEN_LOGITS_FROM
//   pK.acts/    one float32 file per (position, tensor), indexed in pK.acts/index.tsv
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

struct ActCtx {
    std::vector<std::string> prefixes;
    std::string dir;
    int pos = 0;
    int seq = 0; // capture order within the current position
    int max_pos = 4;
    FILE * index = nullptr;
    std::vector<uint8_t> buf;
};

static bool wanted(const ActCtx & a, const char * name) {
    for (const auto & p : a.prefixes) {
        if (strncmp(name, p.c_str(), p.size()) == 0) return true;
    }
    return false;
}

static bool act_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * a = (ActCtx *) ud;
    if (a->pos >= a->max_pos || !wanted(*a, t->name)) return false;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || strstr(t->name, "(reshaped)") || strstr(t->name, "(view)")) return true;
    const size_t n = ggml_nbytes(t);
    a->buf.resize(n);
    ggml_backend_tensor_get(t, a->buf.data(), 0, n);
    char fn[512];
    const int seq = a->seq++;
    snprintf(fn, sizeof fn, "%s/%d_%03d_%s.f32", a->dir.c_str(), a->pos, seq, t->name);
    if (FILE * f = fopen(fn, "wb")) { fwrite(a->buf.data(), 1, n, f); fclose(f); }
    fprintf(a->index, "%d\t%d\t%s\t%lld\t%lld\t%lld\t%lld\t%d_%03d_%s.f32\n", a->pos, seq, t->name,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], a->pos, seq, t->name);
    return true;
}

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: golden MODEL.gguf OUT_DIR PROMPT_FILE...\n"); return 2; }
    const char * model_path = argv[1];
    std::string out = argv[2];
    mkdir(out.c_str(), 0755);

    const int logits_from = getenv("QX_GOLDEN_LOGITS_FROM") ? atoi(getenv("QX_GOLDEN_LOGITS_FROM")) : 0;
    const int threads     = getenv("QX_GOLDEN_THREADS") ? atoi(getenv("QX_GOLDEN_THREADS")) : 16;
    const int ngl         = getenv("QX_GOLDEN_NGL") ? atoi(getenv("QX_GOLDEN_NGL")) : 0;

    ActCtx act;
    act.max_pos = getenv("QX_GOLDEN_ACT_POS") ? atoi(getenv("QX_GOLDEN_ACT_POS")) : 4;
    std::string names = getenv("QX_GOLDEN_ACT") ? getenv("QX_GOLDEN_ACT") :
        "model.input_embed,hc_mixed-,linear_attn_out-,attn_output-,ffn_moe_out-,ffn_shexp_gated-,ffn_out-,l_last-,"
        "ple_embd,ple_gate-,ple_conv_out-,result_norm,indexer_score-,kqv_out-,final_output-,hc_inject-";
    for (size_t s = 0, e; s < names.size(); s = e + 1) {
        e = names.find(',', s); if (e == std::string::npos) e = names.size();
        if (e > s) act.prefixes.push_back(names.substr(s, e - s));
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (getenv("QX_GOLDEN_TOKENIZE_ONLY")) mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (int k = 3; k < argc; ++k) {
        std::string text = read_file(argv[k]);
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        toks.resize(n);
        if (getenv("QX_GOLDEN_TOKENIZE_ONLY")) {
            char tp[600];
            snprintf(tp, sizeof tp, "%s/p%d.tokens", out.c_str(), k - 3);
            FILE * ft = fopen(tp, "wb"); fwrite(toks.data(), sizeof(llama_token), toks.size(), ft); fclose(ft);
            fprintf(stderr, "prompt %d: %d tokens written\n", k - 3, n);
            continue;
        }

        auto cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) ((n + 255) / 256 * 256 + 256);
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        char pdir[512]; snprintf(pdir, sizeof pdir, "%s/p%d.acts", out.c_str(), k - 3);
        mkdir(pdir, 0755);
        act.dir = pdir;
        char ipath[600]; snprintf(ipath, sizeof ipath, "%s/index.tsv", pdir);
        act.index = fopen(ipath, "w");
        if (!getenv("QX_GOLDEN_FA")) {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = GGML_TYPE_F32;
            cp.type_v = GGML_TYPE_F32;
        }
        cp.cb_eval = act_cb;
        cp.cb_eval_user_data = &act;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

        char path[600];
        snprintf(path, sizeof path, "%s/p%d.tokens", out.c_str(), k - 3);
        FILE * ft = fopen(path, "wb"); fwrite(toks.data(), sizeof(llama_token), toks.size(), ft); fclose(ft);
        snprintf(path, sizeof path, "%s/p%d.logits", out.c_str(), k - 3);
        FILE * fl = fopen(path, "wb");

        for (int i = 0; i < n; ++i) {
            act.pos = i;
            act.seq = 0;
            llama_batch b = llama_batch_get_one(&toks[i], 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
            if (i >= logits_from) fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, fl);
            if ((i + 1) % 64 == 0) fprintf(stderr, "prompt %d: %d / %d\n", k - 3, i + 1, n);
        }
        fclose(fl);
        fclose(act.index);
        fprintf(stderr, "prompt %d: %d tokens done\n", k - 3, n);
        llama_free(ctx);
    }
    llama_model_free(model);
    return 0;
}
