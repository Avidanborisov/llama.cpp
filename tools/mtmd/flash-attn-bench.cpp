#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct args_t {
    std::string q_path;
    std::string k_path;
    std::string v_path;
    std::string out_path;
    ggml_type q_type = GGML_TYPE_F32;
    ggml_type k_type = GGML_TYPE_F16;
    ggml_type v_type = GGML_TYPE_F16;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;
    int64_t ne3 = 0;
    int warmup = 2;
    int runs = 7;
    float scale = 1.0f;
};

static std::string get_arg(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

static ggml_type parse_type_arg(const std::string & value) {
    if (value == "f32") return GGML_TYPE_F32;
    if (value == "f16") return GGML_TYPE_F16;
    if (value == "bf16") return GGML_TYPE_BF16;
    throw std::runtime_error("unsupported type: " + value);
}

static args_t parse_args(int argc, char ** argv) {
    args_t args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--q") {
            args.q_path = get_arg(i, argc, argv);
        } else if (arg == "--k") {
            args.k_path = get_arg(i, argc, argv);
        } else if (arg == "--v") {
            args.v_path = get_arg(i, argc, argv);
        } else if (arg == "--out") {
            args.out_path = get_arg(i, argc, argv);
        } else if (arg == "--q-type") {
            args.q_type = parse_type_arg(get_arg(i, argc, argv));
        } else if (arg == "--k-type") {
            args.k_type = parse_type_arg(get_arg(i, argc, argv));
        } else if (arg == "--v-type") {
            args.v_type = parse_type_arg(get_arg(i, argc, argv));
        } else if (arg == "--ne0") {
            args.ne0 = std::stoll(get_arg(i, argc, argv));
        } else if (arg == "--ne1") {
            args.ne1 = std::stoll(get_arg(i, argc, argv));
        } else if (arg == "--ne2") {
            args.ne2 = std::stoll(get_arg(i, argc, argv));
        } else if (arg == "--ne3") {
            args.ne3 = std::stoll(get_arg(i, argc, argv));
        } else if (arg == "--warmup") {
            args.warmup = std::stoi(get_arg(i, argc, argv));
        } else if (arg == "--runs") {
            args.runs = std::stoi(get_arg(i, argc, argv));
        } else if (arg == "--scale") {
            args.scale = std::stof(get_arg(i, argc, argv));
        } else {
            throw std::runtime_error("unknown arg: " + arg);
        }
    }

    if (args.q_path.empty() || args.k_path.empty() || args.v_path.empty() || args.out_path.empty()) {
        throw std::runtime_error("required args: --q --k --v --out --ne0 --ne1 --ne2 --ne3 --scale");
    }
    if (args.ne0 <= 0 || args.ne1 <= 0 || args.ne2 <= 0 || args.ne3 <= 0) {
        throw std::runtime_error("tensor dims must all be > 0");
    }
    return args;
}

static std::vector<uint8_t> read_binary(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    in.seekg(0, std::ios::end);
    const size_t size = (size_t) in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    in.read((char *) data.data(), (std::streamsize) size);
    if (!in) {
        throw std::runtime_error("failed to read " + path);
    }
    return data;
}

static void write_binary(const std::string & path, const void * data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write " + path);
    }
    out.write((const char *) data, (std::streamsize) size);
    if (!out) {
        throw std::runtime_error("failed while writing " + path);
    }
}

static double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n == 0) {
        return 0.0;
    }
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const args_t args = parse_args(argc, argv);

        ggml_backend_load_all();

        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
        if (!backend) {
            throw std::runtime_error("failed to initialize any ggml backend");
        }

        ggml_init_params params = {
            /*.mem_size=*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
            /*.mem_base=*/ nullptr,
            /*.no_alloc=*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("ggml_init failed");
        }

        ggml_tensor * q = ggml_new_tensor_4d(ctx, args.q_type, args.ne0, args.ne1, args.ne2, args.ne3);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, args.k_type, args.ne0, args.ne1, args.ne2, args.ne3);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, args.v_type, args.ne0, args.ne1, args.ne2, args.ne3);
        ggml_set_name(q, "q");
        ggml_set_name(k, "k");
        ggml_set_name(v, "v");

        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, args.scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_set_name(out, "out");

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
        }

        const std::vector<uint8_t> q_bytes = read_binary(args.q_path);
        const std::vector<uint8_t> k_bytes = read_binary(args.k_path);
        const std::vector<uint8_t> v_bytes = read_binary(args.v_path);

        if (q_bytes.size() != ggml_nbytes(q)) {
            throw std::runtime_error("q size mismatch");
        }
        if (k_bytes.size() != ggml_nbytes(k)) {
            throw std::runtime_error("k size mismatch");
        }
        if (v_bytes.size() != ggml_nbytes(v)) {
            throw std::runtime_error("v size mismatch");
        }

        ggml_backend_tensor_set(q, q_bytes.data(), 0, q_bytes.size());
        ggml_backend_tensor_set(k, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_set(v, v_bytes.data(), 0, v_bytes.size());

        for (int i = 0; i < args.warmup; ++i) {
            const ggml_status status = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("warmup failed: ") + ggml_status_to_string(status));
            }
            ggml_backend_synchronize(backend);
        }

        std::vector<double> times_ms;
        times_ms.reserve(args.runs);
        for (int i = 0; i < args.runs; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            const ggml_status status = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("compute failed: ") + ggml_status_to_string(status));
            }
            ggml_backend_synchronize(backend);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = end - start;
            times_ms.push_back(elapsed.count());
        }

        std::vector<float> out_data(ggml_nelements(out));
        ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));
        write_binary(args.out_path, out_data.data(), out_data.size() * sizeof(float));

        const auto [min_it, max_it] = std::minmax_element(times_ms.begin(), times_ms.end());
        const double total_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        const std::string backend_name = ggml_backend_name(backend);

        std::ostringstream json;
        json << std::fixed << std::setprecision(6);
        json << "{\n";
        json << "  \"backend\": \"" << backend_name << "\",\n";
        json << "  \"shape\": [" << args.ne0 << ", " << args.ne1 << ", " << args.ne2 << ", " << args.ne3 << "],\n";
        json << "  \"q_type\": \"" << ggml_type_name(args.q_type) << "\",\n";
        json << "  \"k_type\": \"" << ggml_type_name(args.k_type) << "\",\n";
        json << "  \"v_type\": \"" << ggml_type_name(args.v_type) << "\",\n";
        json << "  \"scale\": " << args.scale << ",\n";
        json << "  \"warmup\": " << args.warmup << ",\n";
        json << "  \"runs\": " << args.runs << ",\n";
        json << "  \"times_ms\": [";
        for (size_t i = 0; i < times_ms.size(); ++i) {
            if (i > 0) json << ", ";
            json << times_ms[i];
        }
        json << "],\n";
        json << "  \"median_ms\": " << median_ms(times_ms) << ",\n";
        json << "  \"min_ms\": " << *min_it << ",\n";
        json << "  \"max_ms\": " << *max_it << ",\n";
        json << "  \"mean_ms\": " << (total_ms / times_ms.size()) << ",\n";
        json << "  \"output_path\": \"" << args.out_path << "\"\n";
        json << "}\n";
        std::cout << json.str();

        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
