#ifndef MTXIPU_CONFIG_CONFIG
#define MTXIPU_CONFIG_CONFIG

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Config {
public:
    bool debug;
    bool verbose;
    unsigned int seed;

    bool permutate;
    bool own_reducer;

    bool model;

    static Config& get() {
        static Config instance;
        return instance;
    };

    json to_json()
    {
        json out;

        out["debug"] = debug;
        out["verbose"] = verbose;
        out["seed"] = seed;
        out["permutate"] = permutate;
        out["own_reducer"] = own_reducer;
        out["model"] = model;

        return out;
    }
};

#endif