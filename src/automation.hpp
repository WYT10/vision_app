#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace vision_app {

struct AutomationTrialState {
    bool ok = false;
    bool active = false;
    std::string session;
    std::string trial_id;
    std::string label;
    std::string image_name;
    std::string image_url;
    int version = 0;
};

struct AutomationResultPayload {
    std::string session;
    std::string trial_id;
    std::string expected_label;
    std::string image_name;
    std::string predicted_label;
    double confidence = 0.0;
    bool match = false;
    bool triggered = false;
    bool model_ok = false;
    double infer_ms = 0.0;
    std::string model_summary;
    std::string client_mode;
    std::string roi_jpg_b64;
};

namespace automation_detail {

struct HttpUrl {
    std::string host;
    int port = 80;
    std::string path = "/";
};

inline bool parse_http_url(const std::string& raw, HttpUrl& out, std::string& err) {
    err.clear();
    out = {};
    const std::string prefix = "http://";
    if (raw.rfind(prefix, 0) != 0) {
        err = "only http:// URLs are supported";
        return false;
    }
    std::string rest = raw.substr(prefix.size());
    const auto slash = rest.find('/');
    std::string host_port = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    out.path = (slash == std::string::npos) ? "/" : rest.substr(slash);
    if (host_port.empty()) {
        err = "empty host in URL";
        return false;
    }
    const auto colon = host_port.rfind(':');
    if (colon != std::string::npos && host_port.find(']') == std::string::npos) {
        out.host = host_port.substr(0, colon);
        try {
            out.port = std::stoi(host_port.substr(colon + 1));
        } catch (...) {
            err = "bad port in URL";
            return false;
        }
    } else {
        out.host = host_port;
        out.port = 80;
    }
    if (out.host.empty()) {
        err = "empty host in URL";
        return false;
    }
    if (out.path.empty()) out.path = "/";
    return true;
}

inline bool connect_socket(const HttpUrl& url, int& fd, std::string& err) {
    fd = -1;
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    struct addrinfo* result = nullptr;
    const std::string port = std::to_string(url.port);
    const int rc = ::getaddrinfo(url.host.c_str(), port.c_str(), &hints, &result);
    if (rc != 0) {
        err = std::string("getaddrinfo failed: ") + gai_strerror(rc);
        return false;
    }

    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        const int s = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (s < 0) continue;
        if (::connect(s, rp->ai_addr, rp->ai_addrlen) == 0) {
            fd = s;
            ::freeaddrinfo(result);
            return true;
        }
        ::close(s);
    }
    ::freeaddrinfo(result);
    err = std::string("connect failed: ") + std::strerror(errno);
    return false;
}

inline bool recv_all(int fd, std::string& response, std::string& err) {
    response.clear();
    std::array<char, 8192> buf{};
    while (true) {
        const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
        if (n == 0) return true;
        if (n < 0) {
            if (errno == EINTR) continue;
            err = std::string("recv failed: ") + std::strerror(errno);
            return false;
        }
        response.append(buf.data(), static_cast<size_t>(n));
    }
}

inline bool send_all(int fd, const std::string& request, std::string& err) {
    size_t off = 0;
    while (off < request.size()) {
        const ssize_t n = ::send(fd, request.data() + off, request.size() - off, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            err = std::string("send failed: ") + std::strerror(errno);
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

inline bool http_request(const std::string& method,
                         const std::string& url_text,
                         const std::string& body,
                         const std::string& content_type,
                         std::string& response_body,
                         std::string& err) {
    HttpUrl url;
    if (!parse_http_url(url_text, url, err)) return false;
    int fd = -1;
    if (!connect_socket(url, fd, err)) return false;

    std::ostringstream req;
    req << method << ' ' << url.path << " HTTP/1.1\r\n"
        << "Host: " << url.host << ':' << url.port << "\r\n"
        << "Connection: close\r\n";
    if (!body.empty()) {
        req << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    }
    req << "\r\n";
    if (!body.empty()) req << body;

    std::string raw;
    const bool ok_send = send_all(fd, req.str(), err);
    const bool ok_recv = ok_send && recv_all(fd, raw, err);
    ::close(fd);
    if (!ok_recv) return false;

    const auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) {
        err = "bad HTTP response";
        return false;
    }
    const std::string headers = raw.substr(0, hdr_end);
    response_body = raw.substr(hdr_end + 4);
    std::istringstream hs(headers);
    std::string status_line;
    std::getline(hs, status_line);
    if (status_line.find(" 200 ") == std::string::npos &&
        status_line.find(" 201 ") == std::string::npos) {
        err = "HTTP request failed: " + status_line;
        return false;
    }
    return true;
}

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::ostringstream oss;
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                    out += oss.str();
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

inline std::string json_field_string(const std::string& key, const std::string& value) {
    return "\"" + json_escape(key) + "\":\"" + json_escape(value) + "\"";
}

inline std::string json_field_bool(const std::string& key, bool value) {
    return "\"" + json_escape(key) + "\":" + (value ? "true" : "false");
}

inline std::string json_field_number(const std::string& key, double value) {
    std::ostringstream oss;
    oss << '"' << json_escape(key) << "\":" << std::fixed << std::setprecision(6) << value;
    return oss.str();
}

inline bool regex_extract_string(const std::string& body, const std::string& key, std::string& out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch m;
    if (!std::regex_search(body, m, re)) return false;
    out = m[1].str();
    return true;
}

inline bool regex_extract_int(const std::string& body, const std::string& key, int& out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(body, m, re)) return false;
    out = std::stoi(m[1].str());
    return true;
}

inline bool regex_extract_bool(const std::string& body, const std::string& key, bool& out) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch m;
    if (!std::regex_search(body, m, re)) return false;
    out = m[1].str() == "true";
    return true;
}

inline std::string base64_encode(const unsigned char* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        const uint32_t a = data[i];
        const uint32_t b = (i + 1 < len) ? data[i + 1] : 0;
        const uint32_t c = (i + 2 < len) ? data[i + 2] : 0;
        const uint32_t triple = (a << 16) | (b << 8) | c;
        out.push_back(tbl[(triple >> 18) & 0x3F]);
        out.push_back(tbl[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? tbl[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? tbl[triple & 0x3F] : '=');
    }
    return out;
}

inline std::string base64_encode_mat_jpg(const cv::Mat& img) {
    if (img.empty()) return {};
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 92})) return {};
    return base64_encode(buf.data(), buf.size());
}

} // namespace automation_detail

inline bool automation_fetch_trial(const std::string& server_root,
                                   AutomationTrialState& out,
                                   std::string& err) {
    std::string body;
    if (!automation_detail::http_request("GET", server_root + "/api/current", {}, "application/json", body, err)) return false;
    out = {};
    automation_detail::regex_extract_bool(body, "ok", out.ok);
    automation_detail::regex_extract_bool(body, "active", out.active);
    automation_detail::regex_extract_string(body, "session", out.session);
    automation_detail::regex_extract_string(body, "trial_id", out.trial_id);
    automation_detail::regex_extract_string(body, "label", out.label);
    automation_detail::regex_extract_string(body, "image_name", out.image_name);
    automation_detail::regex_extract_string(body, "image_url", out.image_url);
    automation_detail::regex_extract_int(body, "version", out.version);
    if (!out.ok) {
        err = "server returned no active trial";
        return false;
    }
    err.clear();
    return true;
}

inline bool automation_post_result(const std::string& server_root,
                                   const AutomationResultPayload& payload,
                                   std::string& response_body,
                                   std::string& err) {
    using namespace automation_detail;
    std::ostringstream body;
    body << '{'
         << json_field_string("session", payload.session) << ','
         << json_field_string("trial_id", payload.trial_id) << ','
         << json_field_string("expected_label", payload.expected_label) << ','
         << json_field_string("image_name", payload.image_name) << ','
         << json_field_string("predicted_label", payload.predicted_label) << ','
         << json_field_number("confidence", payload.confidence) << ','
         << json_field_bool("match", payload.match) << ','
         << json_field_bool("triggered", payload.triggered) << ','
         << json_field_bool("model_ok", payload.model_ok) << ','
         << json_field_number("infer_ms", payload.infer_ms) << ','
         << json_field_string("model_summary", payload.model_summary) << ','
         << json_field_string("client_mode", payload.client_mode) << ','
         << json_field_string("roi_jpg_b64", payload.roi_jpg_b64)
         << '}';
    return http_request("POST", server_root + "/api/result", body.str(), "application/json", response_body, err);
}

inline bool automation_append_manifest(const std::string& path, const AutomationResultPayload& payload, std::string& err) {
    err.clear();
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        err = "cannot open automation manifest: " + path;
        return false;
    }
    out << '{'
        << automation_detail::json_field_string("session", payload.session) << ','
        << automation_detail::json_field_string("trial_id", payload.trial_id) << ','
        << automation_detail::json_field_string("expected_label", payload.expected_label) << ','
        << automation_detail::json_field_string("image_name", payload.image_name) << ','
        << automation_detail::json_field_string("predicted_label", payload.predicted_label) << ','
        << automation_detail::json_field_number("confidence", payload.confidence) << ','
        << automation_detail::json_field_bool("match", payload.match) << ','
        << automation_detail::json_field_bool("triggered", payload.triggered) << ','
        << automation_detail::json_field_bool("model_ok", payload.model_ok) << ','
        << automation_detail::json_field_number("infer_ms", payload.infer_ms) << ','
        << automation_detail::json_field_string("model_summary", payload.model_summary) << ','
        << automation_detail::json_field_string("client_mode", payload.client_mode)
        << "}\n";
    return true;
}

inline std::string automation_encode_roi_jpg(const cv::Mat& img) {
    return automation_detail::base64_encode_mat_jpg(img);
}

} // namespace vision_app
