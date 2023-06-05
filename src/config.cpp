#ifndef MTXIPU_CONFIG_CONFIG
#define MTXIPU_CONFIG_CONFIG

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
};

#endif