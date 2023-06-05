#ifndef MTXIPU_REPORT
#define MTXIPU_REPORT

#include <iostream>
#include <map>

#include <poplar/Engine.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using ::poplar::Engine;
using ::poplar::Graph;

class ExperimentReport
{
public:
    void set_timing(std::string name, double time) {
        timings[name] = time;
    }

    json to_json()
    {
        json out;

        // Add timings
        out["timings"] = json::object();

        auto it = timings.begin();
        while (it != timings.end())
        {
            out["timings"][it->first] = (float)it->second;
            it++;
        }

        return out;
    }

private:
    std::map<std::string, double> timings;
};

class ExperimentReportIPU : public ExperimentReport
{
public:
    ExperimentReportIPU(Engine engine, Graph graph) : engine(std::move(engine)), graph(std::move(graph))
    {
    }

    json to_json()
    {
        json out = ExperimentReport::to_json();

        return out;
    }

    Engine engine;
    Graph graph;
};

#endif