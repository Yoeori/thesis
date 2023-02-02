#ifndef MTXIPU_CONFIG
#define MTXIPU_CONFIG

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