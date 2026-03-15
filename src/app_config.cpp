#include "app_config.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace vision_app
{

    static inline std::string trim(const std::string &s)
    {
        size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])))
            ++a;
        size_t b = s.size();
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
            --b;
        return s.substr(a, b - a);
    }

    static bool parse_bool(const std::string &v)
    {
        return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
    }

    bool load_config_file(const std::string &path, AppOptions &opt)
    {
        std::ifstream in(path);
        if (!in.is_open())
            return false;

        std::string line;
        while (std::getline(in, line))
        {
            line = trim(line);
            if (line.empty() || line[0] == '#')
                continue;

            const auto pos = line.find('=');
            if (pos == std::string::npos)
                continue;

            std::string key = trim(line.substr(0, pos));
            std::string val = trim(line.substr(pos + 1));

            if (key == "device")
                opt.device = val;
            else if (key == "width")
                opt.width = std::stoi(val);
            else if (key == "height")
                opt.height = std::stoi(val);
            else if (key == "fps")
                opt.fps = std::stoi(val);
            else if (key == "fourcc")
                opt.fourcc = val;
            else if (key == "duration_sec")
                opt.duration_sec = std::stoi(val);
            else if (key == "warmup_frames")
                opt.warmup_frames = std::stoi(val);
            else if (key == "probe_only")
                opt.probe_only = parse_bool(val);
            else if (key == "list_only")
                opt.list_only = parse_bool(val);
            else if (key == "headless")
                opt.headless = parse_bool(val);
            else if (key == "show_preview")
                opt.show_preview = parse_bool(val);
            else if (key == "save_csv")
                opt.save_csv = parse_bool(val);
            else if (key == "csv_path")
                opt.csv_path = val;
        }
        return true;
    }

    void print_help()
    {
        std::cout
            << "vision_app\n"
            << "Options:\n"
            << "  --device /dev/video0\n"
            << "  --width 640\n"
            << "  --height 480\n"
            << "  --fps 30\n"
            << "  --fourcc MJPG\n"
            << "  --duration 5\n"
            << "  --warmup 10\n"
            << "  --probe-only\n"
            << "  --list-only\n"
            << "  --headless\n"
            << "  --no-preview\n"
            << "  --save-csv\n"
            << "  --csv-path ./report/camera_test.csv\n"
            << "  --config ./vision_app.conf\n"
            << "  -h, --help\n";
    }

    bool parse_args(int argc, char **argv, AppOptions &opt, std::string &err)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string a = argv[i];

            auto need = [&](const char *name) -> std::string
            {
                if (i + 1 >= argc)
                {
                    err = std::string("missing value for ") + name;
                    return {};
                }
                return argv[++i];
            };

            if (a == "-h" || a == "--help")
            {
                print_help();
                err.clear();
                return false;
            }
            else if (a == "--device")
            {
                opt.device = need("--device");
            }
            else if (a == "--width")
            {
                opt.width = std::stoi(need("--width"));
            }
            else if (a == "--height")
            {
                opt.height = std::stoi(need("--height"));
            }
            else if (a == "--fps")
            {
                opt.fps = std::stoi(need("--fps"));
            }
            else if (a == "--fourcc")
            {
                opt.fourcc = need("--fourcc");
            }
            else if (a == "--duration")
            {
                opt.duration_sec = std::stoi(need("--duration"));
            }
            else if (a == "--warmup")
            {
                opt.warmup_frames = std::stoi(need("--warmup"));
            }
            else if (a == "--probe-only")
            {
                opt.probe_only = true;
            }
            else if (a == "--list-only")
            {
                opt.list_only = true;
            }
            else if (a == "--headless")
            {
                opt.headless = true;
                opt.show_preview = false;
            }
            else if (a == "--no-preview")
            {
                opt.show_preview = false;
            }
            else if (a == "--save-csv")
            {
                opt.save_csv = true;
            }
            else if (a == "--csv-path")
            {
                opt.csv_path = need("--csv-path");
            }
            else if (a == "--config")
            {
                opt.config_path = need("--config");
                load_config_file(opt.config_path, opt);
            }
            else
            {
                err = "unknown argument: " + a;
                return false;
            }
        }
        return true;
    }

} // namespace vision_app