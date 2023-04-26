#ifndef MTXIPU_IPU
#define MTXIPU_IPU

#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>

#include "matrix.hpp"

using ::std::map;
using ::std::optional;
using ::std::string;
using ::std::vector;

using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::Graph;
using ::poplar::TargetType;

optional<Device> getIpuDevice(const unsigned int numIpus = 1)
{
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus))
    {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach())
        {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        }
        else
        {
            std::cout << std::endl
                      << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

optional<Device> getIpuModel(const unsigned int numIpus = 1, const unsigned int tilesPerIpu = 10)
{
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIpus;
    ipuModel.tilesPerIPU = tilesPerIpu;
    return ipuModel.createDevice();
}

auto serialize_graph(const Graph &graph)
{
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
    graphSerOfs.close();
}

#endif