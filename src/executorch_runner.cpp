#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <regex>
#include <memory>
#include <optional>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>

#ifdef PHASE1_WITH_EXECUTORCH
#include <executorch/extension/llm/runner/text_llm_runner.h>
#endif

using json = nlohmann::json;

namespace {

const std::string DEFAULT_MODEL = "models/llama.pte";
const std::string DEFAULT_TOKENIZER = "models/tokenizer.json";
// UTF-8 Devanagari danda (U+0964: "।")
const std::string DEVANAGARI_DANDA = "\xE0\xA5\xA4";

std::string trim_ws(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string normalize_language(const std::string& raw_lang) {
    std::string lang;
    lang.reserve(raw_lang.size());
    for (unsigned char c : raw_lang) {
        if (!std::isspace(c)) {
            lang.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    if (lang == "hinglish" || lang == "hi-latn" || lang == "hi_latn") {
        return "hinglish";
    }
    if (lang == "hi" || lang == "hi-in" || lang == "hi_in") {
        return "hi";
    }
    return "en";
}

std::string sanitize_inline(std::string s, size_t max_chars) {
    for (char& c : s) {
        if (c == '\r' || c == '\n' || c == '\t') c = ' ';
    }
    std::string out;
    out.reserve(s.size());
    bool prev_sp = false;
    for (char c : s) {
        if (c == ' ') {
            if (!prev_sp && !out.empty()) out.push_back(' ');
            prev_sp = true;
        } else {
            out.push_back(c);
            prev_sp = false;
        }
    }
    out = trim_ws(out);
    if (out.size() > max_chars) {
        size_t cut = max_chars;
        while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) {
            --cut;
        }
        out = out.substr(0, cut) + "...";
    }
    return out;
}

std::string extract_host_from_text(const std::string& text) {
    static const std::regex url_re(R"(https?://([A-Za-z0-9.\-]{3,80}))", std::regex::icase);
    std::smatch m;
    if (std::regex_search(text, m, url_re) && m.size() > 1) {
        return m[1].str();
    }
    return "";
}

std::string extract_first_ip(const std::string& text) {
    static const std::regex ip_re(R"((\d{1,3}(?:\.\d{1,3}){3}))");
    std::smatch m;
    if (std::regex_search(text, m, ip_re) && m.size() > 1) {
        return m[1].str();
    }
    return "";
}

bool is_real_ascii_sentence_end(const std::string& text, size_t i) {
    char ch = text[i];
    if (ch != '.' && ch != '!' && ch != '?') return false;
    if (ch == '.') {
        bool prev_alnum = (i > 0) && std::isalnum(static_cast<unsigned char>(text[i - 1]));
        bool next_alnum = (i + 1 < text.size()) && std::isalnum(static_cast<unsigned char>(text[i + 1]));
        // Domain/IP/decimal dots (e.g. sbi-kyc.xyz, 192.168.1.99, 0.99)
        if (prev_alnum && next_alnum) return false;
        // Truncated IP octet at very end of string (e.g. "...from 192.")
        if (i + 1 == text.size() && i > 0 && std::isdigit(static_cast<unsigned char>(text[i - 1]))) {
            // Check if token before '.' is a standalone 1-3 digit number like "192."
            size_t j = i;
            while (j > 0 && std::isdigit(static_cast<unsigned char>(text[j - 1]))) --j;
            if (j == 0 || text[j - 1] == ' ') return false;
        }
        // Leading TLD dot after space (e.g. "the .top TLD")
        if (i > 0 && text[i - 1] == ' ') return false;
    }
    return true;
}

int count_sentences_utf8(const std::string& text) {
    int count = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        if (is_real_ascii_sentence_end(text, i)) {
            ++count;
        } else if (i + 2 < text.size() &&
                   static_cast<unsigned char>(text[i]) == 0xE0 &&
                   static_cast<unsigned char>(text[i + 1]) == 0xA5 &&
                   static_cast<unsigned char>(text[i + 2]) == 0xA4) {
            ++count;
            i += 2;
        }
    }
    return count;
}

size_t find_last_sentence_end_utf8(const std::string& text) {
    size_t best = std::string::npos;
    for (size_t i = 0; i < text.size(); ++i) {
        if (is_real_ascii_sentence_end(text, i)) {
            best = i + 1;
        } else if (i + 2 < text.size() &&
                   static_cast<unsigned char>(text[i]) == 0xE0 &&
                   static_cast<unsigned char>(text[i + 1]) == 0xA5 &&
                   static_cast<unsigned char>(text[i + 2]) == 0xA4) {
            best = i + 3;
            i += 2;
        }
    }
    return best;
}

std::string normalize_utf8_word(const std::string& s) {
    std::string norm;
    norm.reserve(s.size());
    for (unsigned char c : s) {
        if (c >= 0x80) {
            norm.push_back(static_cast<char>(c));
        } else if (std::isalnum(c)) {
            norm.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return norm;
}

std::string clean_generated(const std::string& raw) {
    std::istringstream stream(raw);
    std::string line, cleaned;
    while (std::getline(stream, line)) {
        if (line.find("PyTorchObserver") != std::string::npos) continue;
        if (line.find("tokenizers:") != std::string::npos) continue;
        if (line.find("executorch:") != std::string::npos) continue;
        for (const char* tok : {"<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>",
                                "<|start_header_id|>", "<|end_header_id|>"}) {
            size_t pos;
            while ((pos = line.find(tok)) != std::string::npos) {
                line.erase(pos, std::string(tok).size());
            }
        }
        line = trim_ws(line);
        if (line.empty()) continue;
        if (line == "---" || line.rfind("Case ", 0) == 0) break;
        if (line.rfind("Advisor:", 0) == 0) line = trim_ws(line.substr(8));
        if (line.rfind("Assistant:", 0) == 0) line = trim_ws(line.substr(10));
        if (line.rfind("Summary:", 0) == 0) line = trim_ws(line.substr(8));
        if (line.rfind("Note:", 0) == 0) line = trim_ws(line.substr(5));
        size_t star;
        while ((star = line.find("**")) != std::string::npos) {
            line.erase(star, 2);
        }
        line = trim_ws(line);
        if (line.empty()) continue;
        if (!cleaned.empty()) cleaned += " ";
        cleaned += line;
    }

    if (cleaned.size() > 55) {
        size_t last_end = find_last_sentence_end_utf8(cleaned);
        if (last_end != std::string::npos && last_end >= 35) {
            cleaned = cleaned.substr(0, last_end);
        }
    }

    std::vector<std::string> uniq_sents;
    std::vector<std::string> seen_norm;
    std::string cur;
    for (size_t i = 0; i < cleaned.size(); ++i) {
        bool is_danda = (i + 2 < cleaned.size() &&
                         static_cast<unsigned char>(cleaned[i]) == 0xE0 &&
                         static_cast<unsigned char>(cleaned[i + 1]) == 0xA5 &&
                         static_cast<unsigned char>(cleaned[i + 2]) == 0xA4);
        bool is_ascii_end = is_real_ascii_sentence_end(cleaned, i);
        if (is_danda) {
            cur += DEVANAGARI_DANDA;
            i += 2;
        } else {
            cur.push_back(cleaned[i]);
        }

        if (is_danda || is_ascii_end) {
            std::string s = trim_ws(cur);
            cur.clear();
            if (s.size() < 8) continue;
            std::string norm = normalize_utf8_word(s);
            if (!norm.empty() && std::find(seen_norm.begin(), seen_norm.end(), norm) == seen_norm.end()) {
                seen_norm.push_back(norm);
                uniq_sents.push_back(s);
            }
        }
    }
    if (!trim_ws(cur).empty() && uniq_sents.empty()) {
        uniq_sents.push_back(trim_ws(cur));
    }

    std::string dedup;
    for (size_t i = 0; i < uniq_sents.size() && i < 3; ++i) {
        if (!dedup.empty()) dedup += " ";
        dedup += uniq_sents[i];
    }
    return trim_ws(dedup.empty() ? cleaned : dedup);
}

bool has_phrase_loop(const std::string& text) {
    if (text.size() < 42) return false;
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string w;
    while (iss >> w) {
        std::string lw = normalize_utf8_word(w);
        if (!lw.empty()) words.push_back(lw);
    }
    size_t n = words.size();
    if (n < 6) return false;
    for (size_t k = 2; k <= 8; ++k) {
        if (n >= 2 * k) {
            bool eq = true;
            for (size_t i = 0; i < k; ++i) {
                if (words[n - k + i] != words[n - 2 * k + i]) { eq = false; break; }
            }
            if (eq) return true;
        }
    }
    return false;
}

void split_explanation_and_tips(const std::string& cleaned,
                                std::string& out_explanation,
                                std::vector<std::string>& out_recs) {
    if (cleaned.empty()) return;
    std::regex item_regex(R"((?:^|\s)(?:\d+[\.\)]|[-*])\s+)");
    std::sregex_token_iterator iter(cleaned.begin(), cleaned.end(), item_regex, -1);
    std::sregex_token_iterator end;
    std::vector<std::string> parts;
    for (; iter != end; ++iter) {
        std::string p = trim_ws(*iter);
        if (p.size() >= 10) parts.push_back(p);
    }
    if (parts.size() >= 2 && parts[0].size() >= 25) {
        out_explanation = parts[0];
        for (size_t i = 1; i < parts.size() && out_recs.size() < 3; ++i) {
            out_recs.push_back(parts[i]);
        }
    } else {
        out_explanation = cleaned;
    }
}

// Compact diverse 2-shot cross-domain prompt with UNIQUE incident-anchored starter (~115-240 prompt tokens).
// Exploits 512-token context for full input + top-2 evidence clues across 3 languages ("en", "hi", "hinglish")
// with zero fake brands, IPs, or SSIDs in the examples to prevent induction-head copying & loops.
struct BuiltPrompt {
    std::string prompt;
    std::string answer_prefix;
};

BuiltPrompt build_grounded_prompt(const json& event, const std::string& language) {
    std::string source = event.value("source", "detector");
    std::string threat_cat = event.value("threat_category", event.value("category", "security_threat"));
    int risk_score = event.value("risk_score", 70);
    std::string raw_input = sanitize_inline(
        event.value("raw_input_preview", event.value("raw_input", "")),
        210
    );

    struct Clue { int w; std::string desc; };
    std::vector<Clue> clues;
    if (event.contains("evidence") && event["evidence"].is_array()) {
        for (const auto& ev : event["evidence"]) {
            int w = ev.value("weight", 0);
            std::string desc = sanitize_inline(ev.value("description", ""), 75);
            if (!desc.empty() && w > 0) {
                clues.push_back({w, desc});
            }
        }
    }
    std::sort(clues.begin(), clues.end(), [](const Clue& a, const Clue& b) { return a.w > b.w; });
    std::string evidence_line;
    for (size_t i = 0; i < clues.size() && i < 2; ++i) {
        if (!evidence_line.empty()) evidence_line += "; ";
        evidence_line += clues[i].desc;
    }
    if (evidence_line.empty()) evidence_line = threat_cat;

    bool is_url = (source.find("url") != std::string::npos);
    bool is_net = (source.find("network") != std::string::npos);
    std::string host = extract_host_from_text(raw_input);
    std::string first_ip = extract_first_ip(raw_input);

    std::string prefix;
    std::ostringstream ss;

    if (language == "hinglish") {
        // Two compact, diverse cross-domain Hinglish examples (zero fake brands/IPs/SSIDs)
        ss << "Case 1 | dns_hijack (86/100)\n"
           << "Data: Rogue DNS resolver redirecting router lookups\n"
           << "Note: Ek unknown DNS server router ke traffic ko fake websites par redirect kar raha hai. Turant router ki DNS setting check karke trusted server par reset karein.\n"
           << "---\n"
           << "Case 2 | fake_wifi (89/100)\n"
           << "Data: Unencrypted public hotspot capturing login forms\n"
           << "Note: Aisa khula Wi-Fi hotspot aapke passwords aur login details chura sakta hai. Is asurakshit network se turant disconnect karein aur mobile data use karein.\n"
           << "---\n"
           << "Case 3 | " << threat_cat << " (" << risk_score << "/100)\n"
           << "Data: " << raw_input << "\n"
           << "Clues: " << evidence_line << "\n";

        if (is_url && !host.empty()) {
            prefix = "Website " + host + " ek khatarnak phishing link hai kyunki";
        } else if (is_url) {
            prefix = "Yeh link (risk " + std::to_string(risk_score) + "/100) khatarnak phishing threat hai kyunki";
        } else if (is_net && !first_ip.empty()) {
            prefix = "IP address " + first_ip + " se aa raha network traffic sandigdh hai kyunki";
        } else if (is_net) {
            prefix = "Yeh network activity (risk " + std::to_string(risk_score) + "/100) khatarnak hai kyunki";
        } else if (!host.empty()) {
            prefix = "Is SMS mein diya gaya link " + host + " ek khatarnak fraud scam hai kyunki";
        } else {
            prefix = "Yeh SMS alert (risk " + std::to_string(risk_score) + "/100) ek fraud scam hai kyunki";
        }
        ss << "Note: " << prefix;
    } else if (language == "hi") {
        // Two compact, distinct cross-domain Devanagari Hindi examples (zero fake brands/IPs/SSIDs)
        ss << "Case 1 | dns_hijack (86/100)\n"
           << "Data: Rogue DNS resolver redirecting router lookups\n"
           << "Note: एक अनधिकृत DNS सर्वर राउटर के ट्रैफ़िक को नकली वेबसाइटों पर मोड़ रहा है। राउटर की DNS सेटिंग तुरंत जांचें और सुरक्षित सर्वर पर रीसेट करें।\n"
           << "---\n"
           << "Case 2 | fake_wifi (89/100)\n"
           << "Data: Unencrypted public hotspot capturing login forms\n"
           << "Note: खुला असुरक्षित वाई-फ़ाई हॉटस्पॉट आपके पासवर्ड और लॉगिन विवरण चुरा सकता है। ऐसे नेटवर्क से तुरंत डिस्कनेक्ट करें और मोबाइल डेटा का उपयोग करें।\n"
           << "---\n"
           << "Case 3 | " << threat_cat << " (" << risk_score << "/100)\n"
           << "Data: " << raw_input << "\n"
           << "Clues: " << evidence_line << "\n";

        if (is_url && !host.empty()) {
            prefix = "वेबसाइट " + host + " एक खतरनाक फ़िशिंग लिंक है क्योंकि";
        } else if (is_url) {
            prefix = "यह लिंक (जोखिम " + std::to_string(risk_score) + "/100) एक खतरनाक फ़िशिंग खतरा है क्योंकि";
        } else if (is_net && !first_ip.empty()) {
            prefix = "आईपी पता " + first_ip + " से आ रहा नेटवर्क ट्रैफ़िक संदिग्ध है क्योंकि";
        } else if (is_net) {
            prefix = "यह नेटवर्क गतिविधि (जोखिम " + std::to_string(risk_score) + "/100) खतरनाक है क्योंकि";
        } else if (!host.empty()) {
            prefix = "इस संदेश में दिया गया लिंक " + host + " एक खतरनाक धोखाधड़ी है क्योंकि";
        } else {
            prefix = "यह संदेश अलर्ट (जोखिम " + std::to_string(risk_score) + "/100) एक खतरनाक धोखाधड़ी है क्योंकि";
        }
        ss << "Note: " << prefix;
    } else {
        // Two compact, diverse cross-domain English examples (zero fake brands/IPs/SSIDs)
        ss << "Case 1 | dns_hijack (86/100)\n"
           << "Data: Rogue DNS resolver redirecting router lookups\n"
           << "Note: An unauthorized DNS resolver is steering device traffic toward spoofed login sites. Reset router DNS immediately to a trusted provider.\n"
           << "---\n"
           << "Case 2 | fake_wifi (89/100)\n"
           << "Data: Unencrypted public hotspot capturing login forms\n"
           << "Note: This rogue public hotspot intercepts credentials over an unencrypted captive portal. Disconnect right away and switch to trusted mobile data.\n"
           << "---\n"
           << "Case 3 | " << threat_cat << " (" << risk_score << "/100)\n"
           << "Data: " << raw_input << "\n"
           << "Clues: " << evidence_line << "\n";

        if (is_url && !host.empty()) {
            prefix = "The domain " + host + " is flagged as a high-risk phishing threat because";
        } else if (is_url) {
            prefix = "This URL (risk " + std::to_string(risk_score) + "/100) is dangerous because";
        } else if (is_net && !first_ip.empty()) {
            prefix = "Network traffic from " + first_ip + " is flagged as suspicious because";
        } else if (is_net) {
            prefix = "This network flow (risk " + std::to_string(risk_score) + "/100) is suspicious because";
        } else if (!host.empty()) {
            prefix = "This SMS referencing " + host + " is a high-risk scam because";
        } else {
            prefix = "This SMS alert (risk " + std::to_string(risk_score) + "/100) is a scam because";
        }
        ss << "Note: " << prefix;
    }

    return BuiltPrompt{ss.str(), prefix};
}

#ifdef PHASE1_WITH_EXECUTORCH
json handle_request(
    const json& payload,
    executorch::extension::llm::TextLLMRunner* runner,
    const std::string& model_name
)
#else
json handle_request(const json& payload, void* /*runner*/, const std::string& model_name)
#endif
{
    json event = payload.value("event", json::object());
    std::string language = normalize_language(payload.value("language", "en"));
    std::string risk_lvl = event.value("risk_level", "medium");
    std::string pred = event.value("prediction", "suspicious");
    int risk_score = event.value("risk_score", 50);

    bool is_benign = (risk_score == 0) ||
                     (pred == "no_obvious_indicators" && risk_lvl == "low") ||
                     (pred == "no_configured_anomaly" && risk_lvl == "low");
    if (is_benign) {
        std::string msg;
        if (language == "hinglish") {
            msg = "Is input mein koi scam, phishing ya network attack ka sanket nahi mila (Risk: 0/100). Anjaan links par click na karein aur OTP kabhi share na karein.";
        } else if (language == "hi") {
            msg = "इस इनपुट में कोई स्कैम, फ़िशिंग या नेटवर्क हमले का संकेत नहीं मिला (जोखिम: 0/100)। फिर भी अनजान लिंक पर क्लिक न करें और OTP या पिन कभी साझा न करें।";
        } else {
            msg = "No threat, phishing, or scam indicators were triggered on this input (Risk: 0/100). Stay cautious with unexpected links and never share OTPs or passwords.";
        }
        return json{
            {"explanation", msg},
            {"recommendations", json::array()},
            {"metrics", {
                {"model", model_name},
                {"runtime", "executorch-xnnpack-kleidiai-arm64"},
                {"max_context_len", 512},
                {"quantization", "8da4w_g128_emb4_g32"},
                {"language", language},
                {"inference_ms", 0.0},
                {"ttft_ms", 0.0},
                {"generated_tokens", 0},
                {"decode_tok_per_sec", 0.0},
                {"e2e_tok_per_sec", 0.0},
                {"model_used", false},
                {"fast_path_benign", true}
            }}
        };
    }

    BuiltPrompt bp = build_grounded_prompt(event, language);
    std::string full_prompt = bp.prompt;
    std::string generated_text;
    bool model_success = false;
    bool stopped_for_loop = false;
    int gen_tokens = 0;
    double infer_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_tps = 0.0;
    double e2e_tps = 0.0;

#ifdef PHASE1_WITH_EXECUTORCH
    if (runner) {
        bool stopped_for_stopseq = false;
        try {
            int saved_stdout = dup(1);
            if (saved_stdout >= 0) {
                dup2(2, 1);
            }

            auto t0 = std::chrono::steady_clock::now();
            auto t_first = t0;
            bool got_first = false;
            executorch::extension::llm::GenerationConfig config{};
            config.echo = false;
            config.num_bos = 1;
            config.num_eos = 0;
            if (language == "hi") {
                config.max_new_tokens = 140;
                config.temperature = 0.25f;
            } else if (language == "hinglish") {
                config.max_new_tokens = 110;
                config.temperature = 0.30f;
            } else {
                config.max_new_tokens = 100;
                config.temperature = 0.30f;
            }

            runner->reset();
            auto err = runner->generate(
                full_prompt, config,
                [&generated_text, &gen_tokens, &stopped_for_loop, &stopped_for_stopseq, &t_first, &got_first, runner](const std::string& token) {
                    if (!got_first) {
                        t_first = std::chrono::steady_clock::now();
                        got_first = true;
                    }
                    generated_text += token;
                    gen_tokens += 1;
                    int sent_count = count_sentences_utf8(generated_text);
                    if ((gen_tokens >= 18 && sent_count >= 3) ||
                        (gen_tokens >= 8 && (generated_text.find("\n---") != std::string::npos ||
                                             generated_text.find("\nCase ") != std::string::npos ||
                                             generated_text.find("\nData:") != std::string::npos ||
                                             generated_text.find("\nClues:") != std::string::npos ||
                                             generated_text.find("\n\n") != std::string::npos))) {
                        stopped_for_stopseq = true;
                        runner->stop();
                    } else if (gen_tokens >= 16 && has_phrase_loop(generated_text)) {
                        stopped_for_loop = true;
                        runner->stop();
                    }
                },
                nullptr
            );
            auto t1 = std::chrono::steady_clock::now();
            infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (got_first) {
                ttft_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
                double dec_ms = std::chrono::duration<double, std::milli>(t1 - t_first).count();
                if (dec_ms > 1.0 && gen_tokens > 1) {
                    decode_tps = (static_cast<double>(gen_tokens - 1) * 1000.0) / dec_ms;
                }
            }
            if (infer_ms > 0.0 && gen_tokens > 0) {
                e2e_tps = (static_cast<double>(gen_tokens) * 1000.0) / infer_ms;
                if (decode_tps <= 0.0) decode_tps = e2e_tps;
            }
            if ((err == executorch::runtime::Error::Ok || stopped_for_loop || stopped_for_stopseq) && !generated_text.empty()) {
                model_success = true;
            }

            std::fflush(stdout);
            if (saved_stdout >= 0) {
                dup2(saved_stdout, 1);
                close(saved_stdout);
            }
        } catch (const std::exception& exc) {
            std::cerr << "ExecuTorch C++ inference exception: " << exc.what() << std::endl;
        }
    }
#endif

    std::string raw_cont = generated_text;
    for (const char* stop_s : {"\n---", "\nCase ", "\nData:", "\nClues:", "\nNote:", "\n\n"}) {
        auto pos = raw_cont.find(stop_s);
        if (pos != std::string::npos) raw_cont = raw_cont.substr(0, pos);
    }
    std::string full_cont = (!raw_cont.empty() && raw_cont[0] == ' ') ? (bp.answer_prefix + raw_cont)
                                                                     : (bp.answer_prefix + " " + raw_cont);
    std::string cleaned = (model_success && trim_ws(raw_cont).size() >= 12)
        ? clean_generated(full_cont)
        : "";

    std::string final_explanation;
    std::vector<std::string> final_recommendations;
    if (!cleaned.empty()) {
        split_explanation_and_tips(cleaned, final_explanation, final_recommendations);
    }

    bool llm_produced_text = final_explanation.size() >= 25;
    if (!llm_produced_text) {
        final_explanation = "";
        model_success = false;
    }

    return json{
        {"explanation", final_explanation},
        {"recommendations", final_recommendations},
        {"metrics", {
            {"model", model_name},
            {"runtime", "executorch-xnnpack-kleidiai-arm64"},
            {"max_context_len", 512},
            {"quantization", "8da4w_g128_emb4_g32"},
            {"language", language},
            {"inference_ms", std::round(infer_ms * 10.0) / 10.0},
            {"ttft_ms", std::round(ttft_ms * 10.0) / 10.0},
            {"generated_tokens", gen_tokens},
            {"decode_tok_per_sec", std::round(decode_tps * 10.0) / 10.0},
            {"e2e_tok_per_sec", std::round(e2e_tps * 10.0) / 10.0},
            {"model_used", model_success},
            {"loop_guard_triggered", stopped_for_loop},
            {"prompt_chars", static_cast<int>(full_prompt.size())}
        }}
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = DEFAULT_MODEL;
    std::string tokenizer_path = DEFAULT_TOKENIZER;
    bool daemon_mode = false;
    bool reuse_runner = true;

    if (const char* env_m = std::getenv("EDGESHIELD_EXECUTOR_MODEL_PATH")) model_path = env_m;
    if (const char* env_t = std::getenv("EDGESHIELD_EXECUTOR_TOKENIZER_PATH")) tokenizer_path = env_t;
    if (const char* env_r = std::getenv("EDGESHIELD_EXECUTOR_REUSE_RUNNER")) {
        if (std::string(env_r) == "0" || std::string(env_r) == "false") reuse_runner = false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "--model_path") && i + 1 < argc) model_path = argv[++i];
        else if ((arg == "--tokenizer" || arg == "--tokenizer_path") && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--daemon") daemon_mode = true;
        else if (arg == "--no-reuse") reuse_runner = false;
    }

    auto t_load0 = std::chrono::steady_clock::now();
#ifdef PHASE1_WITH_EXECUTORCH
    int saved_stdout = dup(1);
    if (saved_stdout >= 0) {
        dup2(2, 1);
    }
    std::unique_ptr<executorch::extension::llm::TextLLMRunner> runner;
    bool loaded = false;
    try {
        auto tokenizer = executorch::extension::llm::load_tokenizer(tokenizer_path, nullptr, std::nullopt, 0, 0);
        if (tokenizer) {
            runner = executorch::extension::llm::create_text_llm_runner(model_path, std::move(tokenizer));
            if (runner && runner->load() == executorch::runtime::Error::Ok) {
                loaded = true;
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << "ExecuTorch load exception: " << exc.what() << std::endl;
    }
    std::fflush(stdout);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, 1);
        close(saved_stdout);
    }
    auto* runner_ptr = runner.get();
#else
    void* runner_ptr = nullptr;
    bool loaded = false;
#endif
    auto t_load1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

    if (daemon_mode) {
        json ready_msg = {
            {"ready", loaded},
            {"model_path", model_path},
            {"tokenizer_path", tokenizer_path},
            {"load_ms", std::round(load_ms * 10.0) / 10.0},
            {"reuse_runner", reuse_runner},
            {"max_context_len", 512},
            {"runtime", "executorch-xnnpack-kleidiai-arm64"}
        };
        std::cout << ready_msg.dump() << std::endl;
        std::cout.flush();
        if (!loaded) return 2;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try {
                json req = json::parse(line);
                if (req.value("cmd", "") == "ping") {
                    std::cout << json{{"pong", true}, {"ready", true}}.dump() << std::endl;
                    std::cout.flush();
                    continue;
                }
                json resp = handle_request(req, runner_ptr, "Llama-3.2-1B-XNNPACK-ctx512");
                std::cout << resp.dump() << std::endl;
                std::cout.flush();
            } catch (const std::exception& exc) {
                std::cout << json{
                    {"explanation", ""},
                    {"recommendations", json::array()},
                    {"error", exc.what()},
                    {"metrics", {{"model_used", false}}}
                }.dump() << std::endl;
                std::cout.flush();
            }
        }
        return 0;
    }

    json payload;
    try {
        std::cin >> payload;
    } catch (const std::exception&) {
        std::cout << json{
            {"explanation", ""},
            {"recommendations", json::array()},
            {"metrics", {{"model_used", false}}}
        }.dump() << std::endl;
        return 0;
    }
    json resp = handle_request(payload, runner_ptr, "Llama-3.2-1B-XNNPACK-ctx512");
    std::cout << resp.dump() << std::endl;
    return 0;
}
