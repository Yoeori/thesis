#ifndef MTXIPU_CONFIG_CONFIG
#define MTXIPU_CONFIG_CONFIG

class Config {
public:
    bool debug;
    bool verbose;

    static Config& get() {
        static Config instance;
        return instance;
    };
};

#endif