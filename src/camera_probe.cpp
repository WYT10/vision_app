#include "camera_probe.hpp"

#include <array>
#include <cstdio>
#include <iostream>
#include <regex>
#include <sstream>

namespace vision_app
{

    static bool run_command(const std::string &cmd, std::string &out)
    {
        out.clear();
        std::array<char, 512> buf{};
        FILE *fp = popen(cmd.c_str(), "r");
        if (!fp)
            return false;

        while (fgets(buf.data(), static_cast<int>(buf.size()), fp))
        {
            out += buf.data();
        }

        const int rc = pclose(fp);
        return rc == 0;
    }

    bool probe_camera_modes(const std::string &device, ProbeResult &out, std::string &err)
    {
        out = {};
        out.device = device;

        std::string info;
        {
            std::string cmd = "v4l2-ctl -d " + device + " --all 2>/dev/null";
            run_command(cmd, info);
        }

        {
            std::regex card_re(R"(Card type\s*:\s*(.+))");
            std::regex bus_re(R"(Bus info\s*:\s*(.+))");
            std::smatch m;
            std::istringstream iss(info);
            std::string line;
            while (std::getline(iss, line))
            {
                if (std::regex_search(line, m, card_re))
                    out.card_name = m[1];
                if (std::regex_search(line, m, bus_re))
                    out.bus_info = m[1];
            }
        }

        std::string formats;
        {
            std::string cmd = "v4l2-ctl -d " + device + " --list-formats-ext 2>/dev/null";
            if (!run_command(cmd, formats))
            {
                err = "failed to run v4l2-ctl --list-formats-ext";
                return false;
            }
        }

        std::istringstream iss(formats);
        std::string line;
        CameraMode current;
        bool have_mode = false;

        std::regex pix_re(R"(\[\d+\]:\s+'([^']+)')");
        std::regex size_re(R"(Size:\s+Discrete\s+(\d+)x(\d+))");
        std::regex fps_re(R"(([\d.]+)\s+fps)");

        std::smatch m;
        while (std::getline(iss, line))
        {
            if (std::regex_search(line, m, pix_re))
            {
                if (have_mode && current.width > 0)
                {
                    out.modes.push_back(current);
                }
                current = {};
                current.pixel_format = m[1];
                have_mode = true;
            }
            else if (std::regex_search(line, m, size_re))
            {
                if (have_mode && current.width > 0)
                {
                    out.modes.push_back(current);
                }
                current.width = static_cast<uint32_t>(std::stoul(m[1]));
                current.height = static_cast<uint32_t>(std::stoul(m[2]));
            }
            else if (std::regex_search(line, m, fps_re))
            {
                if (current.width > 0)
                {
                    current.fps_list.push_back(std::stod(m[1]));
                }
            }
        }

        if (have_mode && current.width > 0)
        {
            out.modes.push_back(current);
        }

        if (out.modes.empty())
        {
            err = "no camera modes parsed from v4l2-ctl output";
            return false;
        }

        return true;
    }

    void print_probe_result(const ProbeResult &probe)
    {
        std::cout << "Device   : " << probe.device << "\n";
        std::cout << "Card     : " << probe.card_name << "\n";
        std::cout << "Bus      : " << probe.bus_info << "\n";
        std::cout << "Modes    : " << probe.modes.size() << "\n";

        for (const auto &m : probe.modes)
        {
            std::cout << "  " << m.pixel_format
                      << "  " << m.width << "x" << m.height
                      << "  fps:";
            for (double f : m.fps_list)
            {
                std::cout << " " << f;
            }
            std::cout << "\n";
        }
    }

} // namespace vision_app