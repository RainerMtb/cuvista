// Basic instrumentation profiler by Cherno

// Usage: include this header file somewhere in your code (eg. precompiled header), and then use like:
//
// ChernoProfiler::Get().BeginSession("Session Name");        // Begin session 
// {
//     ChernoTimer timer("Profiled Scope Name");   // Place code like this in scopes you'd like to include in profiling
//     // Code
// }
// ChernoProfiler::Get().EndSession();                        // End Session
//
// You will probably want to macro-fy this, to switch on/off easily and use things like __FUNCSIG__ for the profile name.
//
// Open files with google chrome and "chrome://tracing"

//definition in CoreData
#if PROFILING > 0
#define TIMER(name) ChernoTimer timer__LINE__ (name)
#define TIMER_FUNC() TIMER(__FUNCTION__)
#define TIMER_BEGIN(file) ChernoProfiler::Get().BeginSession(file)
#define TIMER_END() ChernoProfiler::Get().EndSession()
#else
#define TIMER(name)
#define TIMER_FUNC()
#define TIMER_BEGIN(file)
#define TIMER_END()
#endif

#pragma once

#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <thread>
#include <mutex>

struct ProfileResult {
    std::string& Name;
    long long Start, End;
    size_t ThreadID;
};

struct ChernoProfilerSession {
    std::string Name;
};

//---- ChernoProfiler
class ChernoProfiler {

private:
    ChernoProfilerSession* m_CurrentSession;
    std::ofstream m_OutputStream;
    int m_ProfileCount;
    std::mutex m_Mutex;

public:
    ChernoProfiler() : m_CurrentSession(nullptr), m_ProfileCount(0) {}

    static ChernoProfiler& Get() {
        static ChernoProfiler instance;
        return instance;
    }

    void BeginSession(const std::string& filepath = "results.json") {
        m_OutputStream.open(filepath);
        m_OutputStream << "{\"otherData\": {},\"traceEvents\":[";
        m_OutputStream.flush();
        m_CurrentSession = new ChernoProfilerSession { "session" };
    }

    void EndSession() {
        m_OutputStream << "]}";
        m_OutputStream.close();
        delete m_CurrentSession;
        m_CurrentSession = nullptr;
        m_ProfileCount = 0;
    }

    void WriteProfile(const ProfileResult& result) {
        std::stringstream ss;

        std::replace(result.Name.begin(), result.Name.end(), '"', '\'');

        if (m_ProfileCount++ > 0) ss << ",";
        ss << "{";
        ss << "\"cat\":\"function\",";
        ss << "\"dur\":" << (result.End - result.Start) << ',';
        ss << "\"name\":\"" << result.Name << "\",";
        ss << "\"ph\":\"X\",";
        ss << "\"pid\":0,";
        ss << "\"tid\":" << result.ThreadID << ",";
        ss << "\"ts\":" << result.Start;
        ss << "}";

        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            m_OutputStream << ss.str();
            m_OutputStream.flush();
        }
    }
};


//---- main user class
class ChernoTimer {

private:
    std::string m_Name;
    bool m_Stopped;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;

public:
    ChernoTimer(std::string&& name) : m_Name { name }, m_Stopped { false }, m_StartTimepoint { std::chrono::high_resolution_clock::now() } {}

    ChernoTimer(const char* name) : m_Name { name }, m_Stopped { false }, m_StartTimepoint { std::chrono::high_resolution_clock::now() } {}

    ~ChernoTimer() {
        if (!m_Stopped) Stop();
    }

    void Stop() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        size_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
        ChernoProfiler::Get().WriteProfile({ m_Name, start, end, threadID });

        m_Stopped = true;
    }
};
