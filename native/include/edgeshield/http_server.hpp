#pragma once
#include <string>
namespace edgeshield { class HttpServer { public: explicit HttpServer(int port=8080, std::string host="127.0.0.1"); int run(); private: int port_; std::string host_; }; }
