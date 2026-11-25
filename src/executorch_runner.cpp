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
        if (!std::isspace(c)) lang.push_back(static_cast<char>(std::tolower(c)));
    }
    if (lang == "hinglish" || lang == "hi-latn" || lang == "hi_latn") return "hinglish";
    if (lang == "hi" || lang == "hi-in" || lang == "hi_in") return "hi";
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
        while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) --cut;
        out = out.substr(0, cut) + "...";
    }
    return out;
}

std::string extract_host_from_text(const std::string& text) {
    static const std::regex url_re(R"(https?://([A-Za-z0-9.\-]{3,80}))", std::regex::icase);
    std::smatch m;
    if (std::regex_search(text, m, url_re) && m.size() > 1) return m[1].str();
    return "";
}

std::string extract_first_ip(const std::string& text) {
    static const std::regex ip_re(R"((\d{1,3}(?:\.\d{1,3}){3}))");
    std::smatch m;
    if (std::regex_search(text, m, ip_re) && m.size() > 1) return m[1].str();
    return "";
}

bool is_real_ascii_sentence_end(const std::string& text, size_t i) {
    char ch = text[i];
    if (ch != '.' && ch != '!' && ch != '?') return false;
    if (ch == '.') {
        bool prev_alnum = (i > 0) && std::isalnum(static_cast<unsigned char>(text[i - 1]));
        bool next_alnum = (i + 1 < text.size()) && std::isalnum(static_cast<unsigned char>(text[i + 1]));
        if (prev_alnum && next_alnum) return false;
        if (i + 1 == text.size() && i > 0 && std::isdigit(static_cast<unsigned char>(text[i - 1]))) {
            size_t j = i;
            while (j > 0 && std::isdigit(static_cast<unsigned char>(text[j - 1]))) --j;
            if (j == 0 || text[j - 1] == ' ') return false;
        }
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

std::string normalize_utf8_word(const std::string& s) {
    std::string norm;
    norm.reserve(s.size());
    for (unsigned char c : s) {
        if (c >= 0x80) norm.push_back(static_cast<char>(c));
        else if (std::isalnum(c)) norm.push_back(static_cast<char>(std::tolower(c)));
    }
    return norm;
}

std::string clean_paragraph(const std::string& raw, int max_sents = 2) {
    std::string s = raw;
    for (const char* tok : {"<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>",
                            "<|start_header_id|>", "<|end_header_id|>"}) {
        size_t pos;
        while ((pos = s.find(tok)) != std::string::npos) s.erase(pos, std::string(tok).size());
    }
    size_t star;
    while ((star = s.find("**")) != std::string::npos) s.erase(star, 2);

    std::string out;
    out.reserve(s.size());
    bool prev_sp = false;
    for (char c : s) {
        if (c == '\r' || c == '\n' || c == '\t' || c == ' ') {
            if (!prev_sp && !out.empty()) out.push_back(' ');
            prev_sp = true;
        } else {
            out.push_back(c);
            prev_sp = false;
        }
    }
    out = trim_ws(out);
    if (out.size() > 40) {
        size_t best = std::string::npos;
        for (size_t i = 0; i < out.size(); ++i) {
            if (is_real_ascii_sentence_end(out, i)) best = i + 1;
            else if (i + 2 < out.size() &&
                     static_cast<unsigned char>(out[i]) == 0xE0 &&
                     static_cast<unsigned char>(out[i + 1]) == 0xA5 &&
                     static_cast<unsigned char>(out[i + 2]) == 0xA4) {
                best = i + 3;
                i += 2;
            }
        }
        if (best != std::string::npos && best >= 25) out = out.substr(0, best);
    }

    std::vector<std::string> uniq_sents;
    std::vector<std::string> seen_norm;
    std::string cur;
    for (size_t i = 0; i < out.size(); ++i) {
        bool is_danda = (i + 2 < out.size() &&
                         static_cast<unsigned char>(out[i]) == 0xE0 &&
                         static_cast<unsigned char>(out[i + 1]) == 0xA5 &&
                         static_cast<unsigned char>(out[i + 2]) == 0xA4);
        bool is_ascii_end = is_real_ascii_sentence_end(out, i);
        if (is_danda) { cur += DEVANAGARI_DANDA; i += 2; }
        else { cur.push_back(out[i]); }

        if (is_danda || is_ascii_end) {
            std::string sent = trim_ws(cur);
            cur.clear();
            if (sent.size() < 8) continue;
            std::string norm = normalize_utf8_word(sent);
            if (!norm.empty() && std::find(seen_norm.begin(), seen_norm.end(), norm) == seen_norm.end()) {
                seen_norm.push_back(norm);
                uniq_sents.push_back(sent);
            }
        }
    }
    if (!trim_ws(cur).empty() && uniq_sents.empty()) uniq_sents.push_back(trim_ws(cur));

    std::string dedup;
    for (size_t i = 0; i < uniq_sents.size() && static_cast<int>(i) < max_sents; ++i) {
        if (!dedup.empty()) dedup += " ";
        dedup += uniq_sents[i];
    }
    return trim_ws(dedup.empty() ? out : dedup);
}

bool has_phrase_loop(const std::string& text) {
    if (text.size() < 36) return false;
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

struct EventCtx {
    std::string source;
    std::string threat_cat;
    int risk_score;
    std::string raw_input;
    std::string evidence_line;
    std::string host;
    std::string first_ip;
    bool is_url;
    bool is_net;
};

EventCtx parse_event_ctx(const json& event) {
    EventCtx c;
    c.source = event.value("source", "detector");
    c.threat_cat = event.value("threat_category", event.value("category", "security_threat"));
    c.risk_score = event.value("risk_score", 70);
    c.raw_input = sanitize_inline(event.value("raw_input_preview", event.value("raw_input", "")), 185);
    struct Clue { int w; std::string desc; };
    std::vector<Clue> clues;
    if (event.contains("evidence") && event["evidence"].is_array()) {
        for (const auto& ev : event["evidence"]) {
            int w = ev.value("weight", 0);
            std::string desc = sanitize_inline(ev.value("description", ""), 75);
            if (!desc.empty() && w > 0) clues.push_back({w, desc});
        }
    }
    std::sort(clues.begin(), clues.end(), [](const Clue& a, const Clue& b) { return a.w > b.w; });
    for (size_t i = 0; i < clues.size() && i < 2; ++i) {
        if (!c.evidence_line.empty()) c.evidence_line += "; ";
        c.evidence_line += clues[i].desc;
    }
    if (c.evidence_line.empty()) c.evidence_line = c.threat_cat;
    c.host = extract_host_from_text(c.raw_input);
    c.first_ip = extract_first_ip(c.raw_input);
    c.is_url = (c.source.find("url") != std::string::npos);
    c.is_net = (c.source.find("network") != std::string::npos);
    return c;
}

#ifdef PHASE1_WITH_EXECUTORCH
bool run_generation(
    executorch::extension::llm::TextLLMRunner* runner,
    const std::string& prompt,
    int max_new_tokens,
    float temperature,
    int max_sents,
    bool stop_on_list_3,
    std::string& out_text,
    int& out_tokens,
    double& out_ms,
    double& out_ttft_ms,
    bool& out_loop
) {
    out_text.clear();
    out_tokens = 0;
    out_ms = 0.0;
    out_ttft_ms = 0.0;
    out_loop = false;
    if (!runner) return false;

    bool stopped_for_stopseq = false;
    auto t0 = std::chrono::steady_clock::now();
    auto t_first = t0;
    bool got_first = false;

    executorch::extension::llm::GenerationConfig config{};
    config.echo = false;
    config.num_bos = 1;
    config.num_eos = 0;
    config.max_new_tokens = max_new_tokens;
    config.temperature = temperature;

    runner->reset();
    auto err = runner->generate(
        prompt, config,
        [&](const std::string& token) {
            if (!got_first) {
                t_first = std::chrono::steady_clock::now();
                got_first = true;
            }
            out_text += token;
            out_tokens += 1;
            int sc = count_sentences_utf8(out_text);
            if ((max_sents > 0 && out_tokens >= 18 && sc >= max_sents) ||
                (stop_on_list_3 && out_tokens >= 16 && (out_text.find("\n4.") != std::string::npos || out_text.find("\n---") != std::string::npos)) ||
                (out_tokens >= 10 && (out_text.find("\n---") != std::string::npos ||
                                      out_text.find("\nAlert:") != std::string::npos ||
                                      out_text.find("\nTarget:") != std::string::npos ||
                                      out_text.find("\nReason:") != std::string::npos ||
                                      out_text.find("\n\n\n") != std::string::npos))) {
                stopped_for_stopseq = true;
                runner->stop();
            } else if (out_tokens >= 16 && has_phrase_loop(out_text)) {
                out_loop = true;
                runner->stop();
            }
        },
        nullptr
    );
    auto t1 = std::chrono::steady_clock::now();
    out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (got_first) out_ttft_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
    return (err == executorch::runtime::Error::Ok || out_loop || stopped_for_stopseq) && !out_text.empty();
}
#endif

std::string build_explanation_prompt(const EventCtx& c, const std::string& lang, std::string& out_prefix) {
    std::ostringstream ss;
    if (lang == "hi") {
        if (c.is_url && !c.host.empty()) out_prefix = "वेबसाइट " + c.host + " एक खतरनाक फ़िशिंग लिंक है क्योंकि";
        else if (c.is_url) out_prefix = "यह लिंक (जोखिम " + std::to_string(c.risk_score) + "/100) खतरनाक है क्योंकि";
        else if (c.is_net && !c.first_ip.empty()) out_prefix = "आईपी पता " + c.first_ip + " से आ रहा नेटवर्क ट्रैफ़िक संदिग्ध है क्योंकि";
        else if (c.is_net) out_prefix = "यह नेटवर्क गतिविधि (जोखिम " + std::to_string(c.risk_score) + "/100) संदिग्ध है क्योंकि";
        else if (!c.host.empty()) out_prefix = "इस संदेश में दिया गया लिंक " + c.host + " एक धोखाधड़ी (स्कैम) है क्योंकि";
        else out_prefix = "यह SMS संदेश (जोखिम " + std::to_string(c.risk_score) + "/100) एक धोखाधड़ी (स्कैम) है क्योंकि";

        ss << "Alert: security_risk (90/100)\n"
           << "Evidence: Unverified origin; Urgent credential lure\n"
           << "Explanation: यह गतिविधि संदिग्ध है क्योंकि यह बिना सत्यापित स्रोत के संवेदनशील जानकारी मांग रही है। ऐसे प्रयासों का उद्देश्य खाते की सुरक्षा को नुकसान पहुँचाना होता है।\n"
           << "---\n"
           << "साइबर सुरक्षा चेतावनी\n"
           << "श्रेणी: " << c.threat_cat << " (जोखिम: " << c.risk_score << "/100)\n"
           << "विवरण: " << c.raw_input << "\n"
           << "साक्ष्य: " << c.evidence_line << "\n"
           << "सरल हिंदी व्याख्या:\n" << out_prefix;
    } else if (lang == "hinglish") {
        if (c.is_url && !c.host.empty()) out_prefix = "Website " + c.host + " ek khatarnak phishing link hai kyunki";
        else if (c.is_url) out_prefix = "Yeh link (risk " + std::to_string(c.risk_score) + "/100) khatarnak phishing threat hai kyunki";
        else if (c.is_net && !c.first_ip.empty()) out_prefix = "IP address " + c.first_ip + " se aa raha network traffic sandigdh hai kyunki";
        else if (c.is_net) out_prefix = "Yeh network activity (risk " + std::to_string(c.risk_score) + "/100) sandigdh hai kyunki";
        else if (!c.host.empty()) out_prefix = "Is SMS mein diya gaya link " + c.host + " ek fraud scam hai kyunki";
        else out_prefix = "Yeh SMS alert (risk " + std::to_string(c.risk_score) + "/100) ek fraud scam hai kyunki";

        ss << "Alert: security_risk (90/100)\n"
           << "Evidence: Unverified origin; Urgent lure\n"
           << "Hinglish: Yeh activity sandigdh hai kyunki yeh bina verified source ke sensitive details maang rahi hai. Aise attempts ka maqsad account ya data ko nuksan pahunchana hota hai.\n"
           << "---\n"
           << "Security Alert Report\n"
           << "Category: " << c.threat_cat << " (Risk " << c.risk_score << "/100)\n"
           << "Target: " << c.raw_input << "\n"
           << "Evidence: " << c.evidence_line << "\n"
           << "Hinglish Explanation:\n" << out_prefix;
    } else {
        if (c.is_url && !c.host.empty()) out_prefix = "The domain " + c.host + " is flagged as a phishing threat because";
        else if (c.is_url) out_prefix = "This URL (risk " + std::to_string(c.risk_score) + "/100) is dangerous because";
        else if (c.is_net && !c.first_ip.empty()) out_prefix = "Network traffic from " + c.first_ip + " is flagged as suspicious because";
        else if (c.is_net) out_prefix = "This network flow (risk " + std::to_string(c.risk_score) + "/100) is suspicious because";
        else if (!c.host.empty()) out_prefix = "This SMS referencing " + c.host + " is a scam because";
        else out_prefix = "This SMS alert (risk " + std::to_string(c.risk_score) + "/100) is a scam because";

        ss << "Security Alert Report\n"
           << "Category: " << c.threat_cat << " (Risk " << c.risk_score << "/100)\n"
           << "Target: " << c.raw_input << "\n"
           << "Evidence: " << c.evidence_line << "\n"
           << "Plain-Language Explanation:\n" << out_prefix;
    }
    return ss.str();
}

std::string build_actions_prompt(const EventCtx& c, const std::string& lang, std::string& out_step1_prefix) {
    std::ostringstream ss;
    if (lang == "hi") {
        if (c.is_url) out_step1_prefix = "1. इस संदिग्ध लिंक को न खोलें और";
        else if (c.is_net) out_step1_prefix = "1. इस संदिग्ध नेटवर्क ट्रैफ़िक की जांच करें और";
        else out_step1_prefix = "1. इस संदेश पर भरोसा न करें और";

        ss << "उदाहरण उपाय:\n"
           << "1. इस संदिग्ध अनुरोध पर भरोसा न करें और कोई गोपनीय जानकारी साझा न करें।\n"
           << "2. केवल आधिकारिक ऐप या सत्यापित लॉग से स्थिति की जांच करें।\n"
           << "---\n"
           << "सुरक्षा चेतावनी: " << c.threat_cat << " (" << c.risk_score << "/100)\n"
           << "विवरण: " << c.raw_input << "\n"
           << "साक्ष्य: " << c.evidence_line << "\n"
           << "तत्काल सुरक्षा उपाय (हिंदी में 3 कदम):\n" << out_step1_prefix;
    } else if (lang == "hinglish") {
        if (c.is_url) out_step1_prefix = "1. Is suspicious link ko bilkul open na karein aur";
        else if (c.is_net) out_step1_prefix = "1. Is suspicious network traffic ko check karein aur";
        else out_step1_prefix = "1. Is SMS par bharosa na karein aur";

        ss << "Example Steps:\n"
           << "1. Is sandigdh request ko ignore karein aur koi bhi password ya OTP share na karein.\n"
           << "2. Hamesha official app ya trusted logs se khud verify karein.\n"
           << "---\n"
           << "Security Alert: " << c.threat_cat << " (" << c.risk_score << "/100)\n"
           << "Target: " << c.raw_input << "\n"
           << "Evidence: " << c.evidence_line << "\n"
           << "Remediation Actions (Hinglish mein 3 steps):\n" << out_step1_prefix;
    } else {
        if (c.is_url) out_step1_prefix = "1. Do not open this URL or enter";
        else if (c.is_net) out_step1_prefix = "1. Inspect active connections and";
        else out_step1_prefix = "1. Do not reply to this SMS or share";

        ss << "Security Alert: " << c.threat_cat << " (" << c.risk_score << "/100)\n"
           << "Target: " << c.raw_input << "\n"
           << "Evidence: " << c.evidence_line << "\n"
           << "Recommended Remediation Actions (3 concise steps):\n" << out_step1_prefix;
    }
    return ss.str();
}

std::vector<std::string> parse_numbered_steps(const std::string& full_text) {
    std::vector<std::string> steps;
    std::vector<std::string> seen;
    std::istringstream iss(full_text);
    std::string line;
    while (std::getline(iss, line)) {
        line = trim_ws(line);
        if (line.empty()) continue;
        if (line.rfind("---", 0) == 0 || line.rfind("Alert:", 0) == 0 || line.rfind("Target:", 0) == 0) break;
        if (line.size() >= 3 && std::isdigit(static_cast<unsigned char>(line[0])) && (line[1] == '.' || line[1] == ')')) {
            line = trim_ws(line.substr(2));
        } else if (line.size() >= 2 && (line[0] == '-' || line[0] == '*')) {
            line = trim_ws(line.substr(1));
        } else if (steps.empty()) {
            // Continuation of step 1
        } else {
            continue;
        }
        line = clean_paragraph(line, 1);
        if (line.size() < 16) continue;
        std::string norm = normalize_utf8_word(line);
        if (!norm.empty() && std::find(seen.begin(), seen.end(), norm) == seen.end()) {
            seen.push_back(norm);
            steps.push_back(line);
            if (steps.size() >= 3) break;
        }
    }
    return steps;
}

#ifdef PHASE1_WITH_EXECUTORCH
json handle_request(const json& payload, executorch::extension::llm::TextLLMRunner* runner, const std::string& model_name)
#else
json handle_request(const json& payload, void* runner, const std::string& model_name)
#endif
{
    json event = payload.value("event", json::object());
    std::string lang = normalize_language(payload.value("language", "en"));
    std::string risk_lvl = event.value("risk_level", "medium");
    std::string pred = event.value("prediction", "suspicious");
    int risk_score = event.value("risk_score", 50);

    bool is_benign = (risk_score == 0) ||
                     (pred == "no_obvious_indicators" && risk_lvl == "low") ||
                     (pred == "no_configured_anomaly" && risk_lvl == "low");
    if (is_benign) {
        std::string msg;
        if (lang == "hi") msg = "इस इनपुट में कोई स्कैम, फ़िशिंग या नेटवर्क हमले का संकेत नहीं मिला (जोखिम: 0/100)। फिर भी अनजान लिंक पर क्लिक न करें और OTP या पिन कभी साझा न करें।";
        else if (lang == "hinglish") msg = "Is input mein koi scam, phishing ya network attack ka sanket nahi mila (Risk: 0/100). Anjaan links par click na karein aur OTP kabhi share na karein.";
        else msg = "No threat, phishing, or scam indicators were triggered on this input (Risk: 0/100). Stay cautious with unexpected links and never share OTPs or passwords.";
        return json{
            {"explanation", msg},
            {"recommendations", json::array()},
            {"metrics", {
                {"model", model_name}, {"runtime", "executorch-xnnpack-kleidiai-arm64"},
                {"max_context_len", 512}, {"quantization", "8da4w_g128_emb4_g32"}, {"language", lang},
                {"inference_ms", 0.0}, {"ttft_ms", 0.0}, {"generated_tokens", 0},
                {"decode_tok_per_sec", 0.0}, {"e2e_tok_per_sec", 0.0},
                {"model_used", false}, {"fast_path_benign", true}
            }}
        };
    }

    EventCtx ctx = parse_event_ctx(event);
    std::string exp_prefix, act_prefix;
    std::string exp_prompt = build_explanation_prompt(ctx, lang, exp_prefix);
    std::string act_prompt = build_actions_prompt(ctx, lang, act_prefix);

    std::string gen_exp, gen_act;
    bool exp_ok = false, act_ok = false;
    bool loop_exp = false, loop_act = false;
    int tok_exp = 0, tok_act = 0;
    double ms_exp = 0.0, ms_act = 0.0;
    double ttft_exp = 0.0, ttft_act = 0.0;

#ifdef PHASE1_WITH_EXECUTORCH
    if (runner) {
        try {
            int saved_stdout = dup(1);
            if (saved_stdout >= 0) dup2(2, 1);

            exp_ok = run_generation(
                runner, exp_prompt,
                (lang == "hi") ? 85 : 55,
                (lang == "hi") ? 0.25f : 0.28f,
                2, false,
                gen_exp, tok_exp, ms_exp, ttft_exp, loop_exp
            );
            act_ok = run_generation(
                runner, act_prompt,
                (lang == "hi") ? 110 : 75,
                (lang == "hi") ? 0.25f : 0.28f,
                0, true,
                gen_act, tok_act, ms_act, ttft_act, loop_act
            );

            std::fflush(stdout);
            if (saved_stdout >= 0) { dup2(saved_stdout, 1); close(saved_stdout); }
        } catch (const std::exception& exc) {
            std::cerr << "ExecuTorch C++ inference exception: " << exc.what() << std::endl;
        }
    }
#else
    (void)runner;
#endif

    std::string final_explanation;
    if (exp_ok && trim_ws(gen_exp).size() >= 10) {
        std::string full_exp = (!gen_exp.empty() && gen_exp[0] == ' ') ? (exp_prefix + gen_exp) : (exp_prefix + " " + gen_exp);
        final_explanation = clean_paragraph(full_exp, 2);
    }

    std::vector<std::string> final_recommendations;
    if (act_ok && trim_ws(gen_act).size() >= 10) {
        std::string full_act = (!gen_act.empty() && gen_act[0] == ' ') ? (act_prefix + gen_act) : (act_prefix + " " + gen_act);
        final_recommendations = parse_numbered_steps(full_act);
    }

    bool model_success = final_explanation.size() >= 25;
    if (!model_success) {
        final_explanation = "";
        final_recommendations.clear();
    }

    int total_tok = tok_exp + tok_act;
    double total_ms = ms_exp + ms_act;
    double e2e_tps = (total_ms > 0.0 && total_tok > 0) ? (static_cast<double>(total_tok) * 1000.0) / total_ms : 0.0;
    double dec_ms = std::max(1.0, (ms_exp - ttft_exp) + (ms_act - ttft_act));
    double decode_tps = (total_tok > 2) ? (static_cast<double>(total_tok - 2) * 1000.0) / dec_ms : e2e_tps;

    return json{
        {"explanation", final_explanation},
        {"recommendations", final_recommendations},
        {"metrics", {
            {"model", model_name},
            {"runtime", "executorch-xnnpack-kleidiai-arm64"},
            {"max_context_len", 512},
            {"quantization", "8da4w_g128_emb4_g32"},
            {"language", lang},
            {"inference_ms", std::round(total_ms * 10.0) / 10.0},
            {"ttft_ms", std::round(ttft_exp * 10.0) / 10.0},
            {"generated_tokens", total_tok},
            {"decode_tok_per_sec", std::round(decode_tps * 10.0) / 10.0},
            {"e2e_tok_per_sec", std::round(e2e_tps * 10.0) / 10.0},
            {"model_used", model_success},
            {"llm_recommendations_count", static_cast<int>(final_recommendations.size())},
            {"loop_guard_triggered", loop_exp || loop_act},
            {"prompt_chars", static_cast<int>(exp_prompt.size() + act_prompt.size())}
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
    if (saved_stdout >= 0) dup2(2, 1);
    std::unique_ptr<executorch::extension::llm::TextLLMRunner> runner;
    bool loaded = false;
    try {
        auto tokenizer = executorch::extension::llm::load_tokenizer(tokenizer_path, nullptr, std::nullopt, 0, 0);
        if (tokenizer) {
            runner = executorch::extension::llm::create_text_llm_runner(model_path, std::move(tokenizer));
            if (runner && runner->load() == executorch::runtime::Error::Ok) loaded = true;
        }
    } catch (const std::exception& exc) {
        std::cerr << "ExecuTorch load exception: " << exc.what() << std::endl;
    }
    std::fflush(stdout);
    if (saved_stdout >= 0) { dup2(saved_stdout, 1); close(saved_stdout); }
    auto* runner_ptr = runner.get();
#else
    void* runner_ptr = nullptr;
    bool loaded = false;
#endif
    auto t_load1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

    if (daemon_mode) {
        json ready_msg = {
            {"ready", loaded}, {"model_path", model_path}, {"tokenizer_path", tokenizer_path},
            {"load_ms", std::round(load_ms * 10.0) / 10.0}, {"reuse_runner", reuse_runner},
            {"max_context_len", 512}, {"runtime", "executorch-xnnpack-kleidiai-arm64"}
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
                std::cout << json{{"explanation", ""}, {"recommendations", json::array()}, {"error", exc.what()}, {"metrics", {{"model_used", false}}}}.dump() << std::endl;
                std::cout.flush();
            }
        }
        return 0;
    }

    json payload;
    try { std::cin >> payload; }
    catch (const std::exception&) {
        std::cout << json{{"explanation", ""}, {"recommendations", json::array()}, {"metrics", {{"model_used", false}}}}.dump() << std::endl;
        return 0;
    }
    json resp = handle_request(payload, runner_ptr, "Llama-3.2-1B-XNNPACK-ctx512");
    std::cout << resp.dump() << std::endl;
    return 0;
}
