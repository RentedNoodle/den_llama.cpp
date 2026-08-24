#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>

static LONG WINAPI den_crash_handler(EXCEPTION_POINTERS * ep) {
    FILE * f = fopen("C:/Users/james/AppData/Local/Temp/crash_stack.txt", "w");
    if (f) {
        fprintf(f, "EXCEPTION 0x%08X at %p\n", (unsigned)ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress);
        CONTEXT * c0 = ep->ContextRecord;
        fprintf(f, "REGS dst=%p src=%p len=%llu rbx=%p rdx=%p\n", (void*)c0->Rdi, (void*)c0->Rsi, (unsigned long long)c0->Rcx, (void*)c0->Rbx, (void*)c0->Rdx);
        HANDLE proc = GetCurrentProcess();
        SymInitialize(proc, NULL, TRUE);
        STACKFRAME64 sf = {};
        CONTEXT * c = ep->ContextRecord;
        sf.AddrPC.Offset = c->Rip;
        sf.AddrPC.Mode = AddrModeFlat;
        sf.AddrStack.Offset = c->Rsp;
        sf.AddrStack.Mode = AddrModeFlat;
        sf.AddrFrame.Offset = c->Rbp;
        sf.AddrFrame.Mode = AddrModeFlat;
        for (int i = 0; i < 48; i++) {
            if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(), &sf, c, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL)) break;
            DWORD64 addr = sf.AddrPC.Offset;
            char sym[sizeof(SYMBOL_INFO) + 256] = {};
            SYMBOL_INFO * si = (SYMBOL_INFO *) sym;
            si->SizeOfStruct = sizeof(SYMBOL_INFO);
            si->MaxNameLen = 255;
            DWORD64 disp = 0;
            if (SymFromAddr(proc, addr, &disp, si)) {
                fprintf(f, "  #%02d 0x%llX %s+0x%llX\n", i, (unsigned long long)addr, si->Name, (unsigned long long)disp);
            } else {
                fprintf(f, "  #%02d 0x%llX\n", i, (unsigned long long)addr);
            }
        }
        fclose(f);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetUnhandledExceptionFilter(den_crash_handler);
#endif
    return llama_cli(argc, argv);
}
