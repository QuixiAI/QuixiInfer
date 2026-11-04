// Offline routing-trace recorder. Not part of the QuixiInfer build: it links llama.cpp (the DeepSeek-V4.1 PR
// build) and records which experts every token of a corpus selects, per layer, for tools/cache_sim.py.
//
// Build (see make_route_trace.sh). Run:
//   route_trace MODEL.gguf OUT_PREFIX CORPUS.txt
// Writes OUT_PREFIX.tokens (int32 ids) and OUT_PREFIX.trace: for every token, for every layer in order, one row
// of uint16 [layer, id_0 .. id_{k-1}] (k = n_expert_used from the model), token-major.
// Options (environment):
//   QX_TRACE_THREADS=T    CPU threads (default 16)
//   QX_TRACE_NGL=N        layers on the GPU (default 0: CPU only)
//   QX_TRACE_CPU_MOE=1    with NGL > 0, keep the routed experts in host memory (ggml offloads the big ops)
//   QX_TRACE_BATCH=N      n_batch = n_ubatch (default 2048)
//   QX_TRACE_MAX=N        stop after N tokens (default: the whole corpus)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct Trace {
    int n_layer = 0;
    int k = 0;
    FILE * out = nullptr;
    std::vector<int32_t> ids;   // [n_tokens][n_layer][k] for the current ubatch
    std::vector<uint8_t> seen;  // per layer, set when its topk arrived
    int n_tokens = 0;
    long written = 0;
    std::vector<uint8_t> raw;

    void flush() {
        if (n_tokens == 0) return;
        std::vector<uint16_t> row(1 + k);
        for (int t = 0; t < n_tokens; ++t) {
            for (int l = 0; l < n_layer; ++l) {
                if (!seen[l]) { fprintf(stderr, "layer %d missing in ubatch\n", l); exit(1); }
                row[0] = (uint16_t) l;
                for (int j = 0; j < k; ++j) row[1 + j] = (uint16_t) ids[((size_t) t * n_layer + l) * k + j];
                fwrite(row.data(), sizeof(uint16_t), row.size(), out);
            }
        }
        written += n_tokens;
        n_tokens = 0;
        std::fill(seen.begin(), seen.end(), 0);
    }
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * tr = (Trace *) ud;
    const char * name = t->name;
    if (strncmp(name, "ffn_moe_topk-", 13) != 0) return false;
    if (ask) return true;
    const int il = atoi(name + 13);
    if (t->type != GGML_TYPE_I32 || t->ne[0] != tr->k) {
        fprintf(stderr, "unexpected %s: type %d ne0 %lld\n", name, (int) t->type, (long long) t->ne[0]);
        exit(1);
    }
    const int nt = (int) t->ne[1];
    if (tr->n_tokens != 0 && nt != tr->n_tokens) { fprintf(stderr, "token count changed within ubatch\n"); exit(1); }
    tr->n_tokens = nt;
    tr->ids.resize((size_t) nt * tr->n_layer * tr->k);
    // the top-k is a strided view of the argsort: copy its whole extent and pick rows by nb[1]
    const size_t nbytes = ggml_nbytes(t);
    tr->raw.resize(nbytes);
    ggml_backend_tensor_get(t, tr->raw.data(), 0, nbytes);
    for (int tok = 0; tok < nt; ++tok)
        for (int j = 0; j < tr->k; ++j) {
            const size_t off = (size_t) tok * t->nb[1] + (size_t) j * t->nb[0];
            int32_t v;
            memcpy(&v, tr->raw.data() + off, 4);
            tr->ids[((size_t) tok * tr->n_layer + il) * tr->k + j] = v;
        }
    tr->seen[il] = 1;
    return true;
}

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: route_trace MODEL.gguf OUT_PREFIX CORPUS.txt\n"); return 2; }
    const int threads = getenv("QX_TRACE_THREADS") ? atoi(getenv("QX_TRACE_THREADS")) : 16;
    const int ngl     = getenv("QX_TRACE_NGL") ? atoi(getenv("QX_TRACE_NGL")) : 0;
    const int nb      = getenv("QX_TRACE_BATCH") ? atoi(getenv("QX_TRACE_BATCH")) : 2048;
    const long maxtok = getenv("QX_TRACE_MAX") ? atol(getenv("QX_TRACE_MAX")) : 0;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    std::vector<llama_model_tensor_buft_override> ov;
    if (ngl > 0 && getenv("QX_TRACE_CPU_MOE")) {
        ov.push_back({ "\\.ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() });
        ov.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = ov.data();
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    Trace tr;
    tr.n_layer = llama_model_n_layer(model);
    {
        char v[32];
        if (llama_model_meta_val_str(model, "deepseek41.expert_used_count", v, sizeof v) < 0) {
            fprintf(stderr, "no expert_used_count\n"); return 1;
        }
        tr.k = atoi(v);
    }
    tr.seen.assign(tr.n_layer, 0);
    fprintf(stderr, "%d layers, %d experts per token\n", tr.n_layer, tr.k);

    std::string text = read_file(argv[3]);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    if (maxtok > 0 && n > maxtok) { toks.resize(maxtok); n = (int) maxtok; }
    fprintf(stderr, "%d tokens\n", n);

    std::string prefix = argv[2];
    { FILE * f = fopen((prefix + ".tokens").c_str(), "wb"); fwrite(toks.data(), 4, toks.size(), f); fclose(f); }
    tr.out = fopen((prefix + ".trace").c_str(), "wb");

    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((n + 255) / 256 * 256 + 256);
    cp.n_batch = nb;
    cp.n_ubatch = nb;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &tr;
    cp.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    for (int i = 0; i < n; i += nb) {
        const int m = std::min(nb, n - i);
        llama_batch b = llama_batch_get_one(&toks[i], m);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
        tr.flush();
        fprintf(stderr, "%ld / %d tokens traced\n", tr.written, n);
        fflush(tr.out);
    }
    fclose(tr.out);
    fprintf(stderr, "wrote %s.trace (%ld tokens x %d layers x %d)\n", prefix.c_str(), tr.written, tr.n_layer, tr.k);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
