package main

/*
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef int  (*S2StreamingWriteCallback)(const uint8_t* data, int32_t size, void* user_data);
typedef void (*S2StreamingDoneCallback)(void* user_data);
typedef void (*S2StreamingErrorCallback)(const char* message, void* user_data);
typedef int  (*S2StreamingCancelCallback)(void* user_data);

struct S2StreamingCallbacks {
	S2StreamingWriteCallback  on_wav_chunk;
	S2StreamingDoneCallback   on_done;
	S2StreamingErrorCallback  on_error;
	S2StreamingCancelCallback is_cancelled;
	void*                     user_data;
};

struct S2StreamingParams {
	int32_t stream_decode_stride_frames;
	int32_t stream_holdback_frames;
	int32_t codec_decode_context_frames;
	int     low_latency;
	int     segment_sentences;
	int32_t sentence_pause_ms;
	int32_t segment_max_chars;
	const char* voice;
	const char* voice_dir;
};

static void* s2_dlopen(const char* path) {
	return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
}
static void* s2_dlsym(void* h, const char* name) {
	return dlsym(h, name);
}
static const char* s2_dlerror(void) {
	return dlerror();
}

static void*    w_alloc(void* f)
{ return ((void*(*)())f)(); }

static void     w_release(void* f, void* p)
{ ((void(*)(void*))f)(p); }

static void     w_set_log_level(void* f, int32_t v)
{ ((void(*)(int32_t))f)(v); }

static int32_t  w_get_log_level(void* f)
{ return ((int32_t(*)())f)(); }

static void     w_sync_tokenizer(void* f, void* m, void* t)
{ ((void(*)(void*,void*))f)(m,t); }

static int w_init_pipeline(void* f, void* p, void* t, void* m, void* c)
{ return ((int(*)(void*,void*,void*,void*))f)(p,t,m,c); }

static int w_init_pipeline_files(void* f, void* p,
	const char* a, const char* b, int32_t c, int32_t d, int32_t e, int32_t g)
{ return ((int(*)(void*,const char*,const char*,int32_t,int32_t,int32_t,int32_t))f)(p,a,b,c,d,e,g); }

static int w_init_gen_params(void* f, void* p,
	int32_t a, float b, float c, int32_t d, int32_t e, int32_t g, int h)
{ return ((int(*)(void*,int32_t,float,float,int32_t,int32_t,int32_t,int))f)(p,a,b,c,d,e,g,h); }

static int w_init_model(void* f, void* m, const char* a, int32_t b, int32_t c)
{ return ((int(*)(void*,const char*,int32_t,int32_t))f)(m,a,b,c); }

static int w_init_model_gpu(void* f, void* m, const char* a, int32_t b, int32_t c, int32_t d)
{ return ((int(*)(void*,const char*,int32_t,int32_t,int32_t))f)(m,a,b,c,d); }

static int w_init_tokenizer(void* f, void* t, const char* p)
{ return ((int(*)(void*,const char*))f)(t,p); }

static int w_init_codec(void* f, void* c, const char* a, int32_t b, int32_t d)
{ return ((int(*)(void*,const char*,int32_t,int32_t))f)(c,a,b,d); }

static int w_init_prompt(void* f, void* p, int32_t t, const char* a, void* c, int32_t* tp)
{ return ((int(*)(void*,int32_t,const char*,void*,int32_t*))f)(p,t,a,c,tp); }

static void*    w_alloc_buffer(void* f, int s)
{ return ((void*(*)(int))f)(s); }

static float*   w_get_buffer_data(void* f, void* b)
{ return ((float*(*)(void*))f)(b); }

static int w_synthesize(void* f, void* p, void* g, void* a, void* c,
	int32_t* tp, const char* ra, const char* rt, const char* t, const char* o, int32_t* ol)
{ return ((int(*)(void*,void*,void*,void*,int32_t*,const char*,const char*,const char*,const char*,int32_t*))f)(p,g,a,c,tp,ra,rt,t,o,ol); }

static int w_stream(void* f, void* p, void* g, struct S2StreamingCallbacks* cb,
	void* c, int32_t* tp, const char* ra, const char* rt, const char* t, int32_t s)
{ return ((int(*)(void*,void*,struct S2StreamingCallbacks*,void*,int32_t*,const char*,const char*,const char*,int32_t))f)(p,g,cb,c,tp,ra,rt,t,s); }

static int w_stream_ex(void* f, void* p, void* g, struct S2StreamingCallbacks* cb,
	void* c, int32_t* tp, const char* ra, const char* rt, const char* t, struct S2StreamingParams* sp)
{ return ((int(*)(void*,void*,struct S2StreamingCallbacks*,void*,int32_t*,const char*,const char*,const char*,struct S2StreamingParams*))f)(p,g,cb,c,tp,ra,rt,t,sp); }

extern int  s2StreamOnWrite(uint8_t* data, int32_t size, void* user_data);
extern void s2StreamOnDone(void* user_data);
extern void s2StreamOnError(char* message, void* user_data);
extern int  s2StreamIsCancelled(void* user_data);

static float s2_read_float(const float* arr, int32_t idx) { return arr[idx]; }

static void build_streaming_callbacks(struct S2StreamingCallbacks* cb, void* ud) {
	cb->on_wav_chunk = (S2StreamingWriteCallback)s2StreamOnWrite;
	cb->on_done      = (S2StreamingDoneCallback)s2StreamOnDone;
	cb->on_error     = (S2StreamingErrorCallback)s2StreamOnError;
	cb->is_cancelled = (S2StreamingCancelCallback)s2StreamIsCancelled;
	cb->user_data    = ud;
}
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

type runMode int

const (
	modeSmoke        runMode = iota
	modeFromFiles
	modeModular
	modeLegacyStream
	modeStreamEx
)

type backendType int32

const (
	backendCPU    backendType = -1
	backendVulkan backendType = 0
	backendCUDA   backendType = 1
	backendMetal  backendType = 2
)

type options struct {
	libraryPath      string
	mode             runMode
	modelPath        string
	tokenizerPath    string
	text             string
	outputPath       string
	refAudioPath     string
	refText          string
	voice            string
	voiceDir         string
	backend          backendType
	gpuDevice        int32
	gpuLayers        int32
	codecFollow      bool
	maxTokens        int32
	temperature      float32
	topP             float32
	topK             int32
	minTokens        int32
	threads          int32
	verbose          bool
	logLevel         int32
	streamStride     int32
	streamHoldback   int32
	codecContext     int32
	lowLatency       bool
	segmentSentences bool
	sentencePauseMs  int32
	segmentMaxChars  int32
	play             bool
}

func defaultOptions() *options {
	return &options{
		mode:          modeModular,
		text:          "This is a short Go export API test.",
		backend:       backendCPU,
		gpuDevice:     -1,
		gpuLayers:     -1,
		codecFollow:   true,
		maxTokens:     -1,
		temperature:   -1,
		topP:          -1,
		topK:          -1,
		minTokens:     -1,
		threads:       -1,
		logLevel:      0,
		streamHoldback: -1,
		codecContext:  -1,
		sentencePauseMs: 180,
	}
}

func parseArgs(args []string) *options {
	opts := defaultOptions()

	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch arg {
		case "--library":
			opts.libraryPath = nextArg(args, &i, arg)
		case "--mode":
			opts.mode = parseMode(nextArg(args, &i, arg))
		case "--model":
			opts.modelPath = absPath(nextArg(args, &i, arg))
		case "--tokenizer":
			opts.tokenizerPath = absPath(nextArg(args, &i, arg))
		case "--text":
			opts.text = nextArg(args, &i, arg)
		case "--output":
			opts.outputPath = absPath(nextArg(args, &i, arg))
		case "--reference-audio":
			opts.refAudioPath = absPath(nextArg(args, &i, arg))
		case "--reference-text":
			opts.refText = nextArg(args, &i, arg)
		case "--voice":
			opts.voice = nextArg(args, &i, arg)
		case "--voice-dir":
			opts.voiceDir = absPath(nextArg(args, &i, arg))
		case "--backend":
			opts.backend = parseBackend(nextArg(args, &i, arg))
		case "--gpu-device":
			opts.gpuDevice = parseInt32(nextArg(args, &i, arg))
		case "--gpu-layers":
			opts.gpuLayers = parseInt32(nextArg(args, &i, arg))
		case "--codec-cpu":
			opts.codecFollow = false
		case "--max-tokens":
			opts.maxTokens = parseInt32(nextArg(args, &i, arg))
		case "--temperature":
			opts.temperature = parseFloat32(nextArg(args, &i, arg))
		case "--top-p":
			opts.topP = parseFloat32(nextArg(args, &i, arg))
		case "--top-k":
			opts.topK = parseInt32(nextArg(args, &i, arg))
		case "--min-tokens-before-end":
			opts.minTokens = parseInt32(nextArg(args, &i, arg))
		case "--threads":
			opts.threads = parseInt32(nextArg(args, &i, arg))
		case "--verbose":
			opts.verbose = true
		case "--log-level":
			opts.logLevel = parseLogLevel(nextArg(args, &i, arg))
		case "--stream-decode-stride":
			opts.streamStride = parseInt32(nextArg(args, &i, arg))
		case "--stream-holdback-frames":
			opts.streamHoldback = parseInt32(nextArg(args, &i, arg))
		case "--codec-context-frames":
			opts.codecContext = parseInt32(nextArg(args, &i, arg))
		case "--low-latency":
			opts.lowLatency = true
		case "--segment-sentences":
			opts.segmentSentences = true
		case "--sentence-pause-ms":
			opts.sentencePauseMs = parseInt32(nextArg(args, &i, arg))
		case "--segment-max-chars":
			opts.segmentMaxChars = parseInt32(nextArg(args, &i, arg))
		case "--play":
			opts.play = true
		case "--help", "-h":
			printHelp()
		default:
			fmt.Fprintf(os.Stderr, "Unknown argument: %s\n", arg)
			os.Exit(1)
		}
	}
	return opts
}

func nextArg(args []string, i *int, flag string) string {
	if *i+1 >= len(args) {
		fmt.Fprintf(os.Stderr, "Missing value for %s\n", flag)
		os.Exit(1)
	}
	*i++
	return args[*i]
}

func absPath(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return abs
}

func parseMode(s string) runMode {
	switch strings.ToLower(s) {
	case "smoke":
		return modeSmoke
	case "from-files":
		return modeFromFiles
	case "modular":
		return modeModular
	case "legacy-stream":
		return modeLegacyStream
	case "stream-ex":
		return modeStreamEx
	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\n", s)
		os.Exit(1)
		return 0
	}
}

func parseBackend(s string) backendType {
	switch strings.ToLower(s) {
	case "cpu":
		return backendCPU
	case "vulkan":
		return backendVulkan
	case "cuda":
		return backendCUDA
	case "metal":
		return backendMetal
	default:
		fmt.Fprintf(os.Stderr, "Unknown backend: %s\n", s)
		os.Exit(1)
		return 0
	}
}

func parseLogLevel(s string) int32 {
	switch strings.ToLower(s) {
	case "0", "error":
		return 0
	case "1", "warn", "warning":
		return 1
	case "2", "info":
		return 2
	case "3", "debug":
		return 3
	default:
		fmt.Fprintf(os.Stderr, "Unknown log level: %s\n", s)
		os.Exit(1)
		return 0
	}
}

func parseInt32(s string) int32 {
	var v int32
	fmt.Sscanf(s, "%d", &v)
	return v
}

func parseFloat32(s string) float32 {
	var v float32
	fmt.Sscanf(s, "%f", &v)
	return v
}

func (o *options) validateForSynthesis() {
	if o.modelPath == "" {
		fatal("--model is required")
	}
	if o.tokenizerPath == "" {
		fatal("--tokenizer is required")
	}
	if o.outputPath != "" {
		if dir := filepath.Dir(o.outputPath); dir != "" {
			os.MkdirAll(dir, 0o755)
		}
	}
}

func printHelp() {
	fmt.Print(`Usage:
  go run ./examples/golang \
    --mode <smoke|from-files|modular|legacy-stream|stream-ex> \
    --library <path/to/libs2.so> \
    [other options]

Common options:
  --model <model.gguf>
  --tokenizer <tokenizer.json>
  --text "Text to synthesize"
  --output <path.wav>
  --reference-audio <ref.wav|ref.mp3>
  --reference-text "Transcript for the reference audio"
  --voice <voice-id-or-path.s2voice>
  --voice-dir <voice profile directory>
  --backend <cpu|vulkan|cuda|metal>
  --gpu-device <index>
  --gpu-layers <count>
  --codec-cpu
  --threads <count>
  --max-tokens <count>
  --log-level <error|warning|info|debug>

Streaming-only options:
  --stream-decode-stride <frames>
  --stream-holdback-frames <frames>
  --codec-context-frames <frames>
  --low-latency
  --segment-sentences
  --sentence-pause-ms <milliseconds>
  --segment-max-chars <chars>
  --play   play audio in realtime via aplay (no output file needed)
`)
	os.Exit(0)
}

type s2 struct {
	handle unsafe.Pointer

	allocPipeline   unsafe.Pointer
	releasePipeline unsafe.Pointer
	setLogLevel     unsafe.Pointer
	getLogLevel     unsafe.Pointer
	syncTokenizer   unsafe.Pointer
	initPipeline    unsafe.Pointer
	initPipeFiles   unsafe.Pointer
	allocGenParams  unsafe.Pointer
	releaseGenParams unsafe.Pointer
	initGenParams   unsafe.Pointer
	allocModel      unsafe.Pointer
	releaseModel    unsafe.Pointer
	initModel       unsafe.Pointer
	initModelGPU    unsafe.Pointer
	allocTokenizer  unsafe.Pointer
	releaseTokenizer unsafe.Pointer
	initTokenizer   unsafe.Pointer
	allocCodec      unsafe.Pointer
	releaseCodec    unsafe.Pointer
	initCodec       unsafe.Pointer
	allocCodes      unsafe.Pointer
	releaseCodes    unsafe.Pointer
	initCodes       unsafe.Pointer
	allocBuffer     unsafe.Pointer
	releaseBuffer   unsafe.Pointer
	getBufferData   unsafe.Pointer
	synthesize      unsafe.Pointer
	stream          unsafe.Pointer
	streamEx        unsafe.Pointer
}

func loadS2(path string) (*s2, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))

	h := C.s2_dlopen(cp)
	if h == nil {
		errStr := C.GoString(C.s2_dlerror())
		return nil, fmt.Errorf("dlopen(%s): %s", path, errStr)
	}

	lib := &s2{handle: h}

	syms := map[*unsafe.Pointer]string{
		&lib.allocPipeline:    "AllocS2Pipeline",
		&lib.releasePipeline:  "ReleaseS2Pipeline",
		&lib.setLogLevel:      "SetS2LogLevel",
		&lib.getLogLevel:      "GetS2LogLevel",
		&lib.syncTokenizer:    "SyncS2TokenizerConfigFromS2Model",
		&lib.initPipeline:     "InitializeS2Pipeline",
		&lib.initPipeFiles:    "InitializeS2PipelineFromFiles",
		&lib.allocGenParams:   "AllocS2GenerateParams",
		&lib.releaseGenParams: "ReleaseS2GenerateParams",
		&lib.initGenParams:    "InitializeS2GenerateParams",
		&lib.allocModel:       "AllocS2Model",
		&lib.releaseModel:     "ReleaseS2Model",
		&lib.initModel:        "InitializeS2Model",
		&lib.initModelGPU:     "InitializeS2ModelWithGpuLayers",
		&lib.allocTokenizer:   "AllocS2Tokenizer",
		&lib.releaseTokenizer: "ReleaseS2Tokenizer",
		&lib.initTokenizer:    "InitializeS2Tokenizer",
		&lib.allocCodec:       "AllocS2AudioCodec",
		&lib.releaseCodec:     "ReleaseS2AudioCodec",
		&lib.initCodec:        "InitializeS2AudioCodec",
		&lib.allocCodes:       "AllocS2AudioPromptCodes",
		&lib.releaseCodes:     "ReleaseS2AudioPromptCodes",
		&lib.initCodes:        "InitializeAudioPromptCodes",
		&lib.allocBuffer:      "AllocS2AudioBuffer",
		&lib.releaseBuffer:    "ReleaseS2AudioBuffer",
		&lib.getBufferData:    "GetS2AudioBufferDataPointer",
		&lib.synthesize:       "S2Synthesize",
		&lib.stream:           "S2SynthesizeStreaming",
		&lib.streamEx:         "S2SynthesizeStreamingEx",
	}

	for ptr, name := range syms {
		cn := C.CString(name)
		sym := C.s2_dlsym(h, cn)
		C.free(unsafe.Pointer(cn))
		if sym == nil {
			C.dlclose(h)
			errStr := C.GoString(C.s2_dlerror())
			return nil, fmt.Errorf("dlsym(%s): %s", name, errStr)
		}
		*ptr = unsafe.Pointer(sym)
	}

	return lib, nil
}

func (lib *s2) close() {
	if lib.handle != nil {
		C.dlclose(lib.handle)
		lib.handle = nil
	}
}

func cstring(s string) *C.char {
	if s == "" {
		return nil
	}
	return C.CString(s)
}

var sessionSeq int64

var activeSessions sync.Map

type streamingSession struct {
	id         int64
	cHandle    unsafe.Pointer
	output     *os.File
	outputPath string
	chunkCount int
	byteCount  int
	done       bool
	errors     []string
	mu         sync.Mutex
	play       bool
	player     io.WriteCloser
	playerCmd  *exec.Cmd
	headerSent bool
}

func newStreamingSession(outputPath string, play bool) *streamingSession {
	s := &streamingSession{
		id:         atomic.AddInt64(&sessionSeq, 1),
		outputPath: outputPath,
		play:       play,
	}
	s.cHandle = C.malloc(8)
	*(*int64)(s.cHandle) = s.id
	if outputPath != "" {
		f, err := os.Create(outputPath)
		if err != nil {
			fatal("create output: %v", err)
		}
		s.output = f
	}
	if play {
		s.startPlayer()
	}
	return s
}

func (s *streamingSession) startPlayer() {
	s.playerCmd = exec.Command("aplay", "-q", "-f", "S16_LE", "-r", "44100", "-c", "1")
	var err error
	s.player, err = s.playerCmd.StdinPipe()
	if err != nil {
		fatal("aplay pipe: %v", err)
	}
	if err := s.playerCmd.Start(); err != nil {
		fatal("aplay start: %v (install alsa-utils)", err)
	}
}

func (s *streamingSession) register() unsafe.Pointer {
	activeSessions.Store(s.id, s)
	return s.cHandle
}

func (s *streamingSession) unregister() {
	activeSessions.Delete(s.id)
}

func (s *streamingSession) finalize() {
	if s.output == nil {
		return
	}
	s.output.Sync()
	if s.done && s.byteCount >= 44 {
		patchWavHeader(s.output, s.byteCount)
	}
}

func (s *streamingSession) close() {
	if s.player != nil {
		s.player.Close()
		s.player = nil
	}
	if s.playerCmd != nil && s.playerCmd.Process != nil {
		done := make(chan struct{})
		go func() {
			s.playerCmd.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			s.playerCmd.Process.Kill()
		}
		s.playerCmd = nil
	}
	if s.cHandle != nil {
		C.free(s.cHandle)
		s.cHandle = nil
	}
	if s.output != nil {
		s.output.Close()
	}
}

func sessionFromC(userData unsafe.Pointer) *streamingSession {
	id := *(*int64)(userData)
	v, ok := activeSessions.Load(id)
	if !ok {
		return nil
	}
	return v.(*streamingSession)
}

//export s2StreamOnWrite
func s2StreamOnWrite(data *C.uint8_t, size C.int32_t, userData unsafe.Pointer) C.int {
	s := sessionFromC(userData)
	if s == nil {
		return 0
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.chunkCount++
	s.byteCount += int(size)

	goBytes := C.GoBytes(unsafe.Pointer(data), C.int(size))

	if s.output != nil {
		s.output.Write(goBytes)
	}

	if s.player != nil {
	pcm := goBytes
		if !s.headerSent {
			if len(pcm) <= 44 {
				s.headerSent = true
				return 1
			}
			pcm = pcm[44:]
			s.headerSent = true
		}
		if len(pcm) > 0 {
			s.player.Write(pcm)
		}
	}

	return 1
}

//export s2StreamOnDone
func s2StreamOnDone(userData unsafe.Pointer) {
	s := sessionFromC(userData)
	if s != nil {
		s.mu.Lock()
		s.done = true
		s.mu.Unlock()
	}
}

//export s2StreamOnError
func s2StreamOnError(msg *C.char, userData unsafe.Pointer) {
	s := sessionFromC(userData)
	if s != nil {
		text := C.GoString(msg)
		s.mu.Lock()
		s.errors = append(s.errors, text)
		s.mu.Unlock()
	}
}

//export s2StreamIsCancelled
func s2StreamIsCancelled(userData unsafe.Pointer) C.int {
	return 0
}

func patchWavHeader(f *os.File, totalSize int) {
	if totalSize < 44 {
		return
	}

	riffSize := uint32(totalSize - 8)
	dataSize := uint32(totalSize - 44)

	cursor, _ := f.Seek(0, io.SeekCurrent)

	f.Seek(4, io.SeekStart)
	binary.Write(f, binary.LittleEndian, riffSize)

	f.Seek(40, io.SeekStart)
	binary.Write(f, binary.LittleEndian, dataSize)

	f.Seek(cursor, io.SeekStart)
	f.Sync()
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func requireHandle(name string, ptr unsafe.Pointer) unsafe.Pointer {
	if ptr == nil {
		fatal("%s returned NULL", name)
	}
	return ptr
}

func expectSuccess(result C.int, msg string) {
	if result != 1 {
		fatal("%s: return_code=%d", msg, int(result))
	}
}

func releaseHandle(ptr unsafe.Pointer, releaseFn unsafe.Pointer) {
	if ptr != nil && releaseFn != nil {
		C.w_release(releaseFn, ptr)
	}
}

func initGenParams(lib *s2, opts *options) unsafe.Pointer {
	p := requireHandle("AllocS2GenerateParams", C.w_alloc(lib.allocGenParams))
	rc := C.w_init_gen_params(lib.initGenParams, p,
		C.int32_t(opts.maxTokens), C.float(opts.temperature), C.float(opts.topP),
		C.int32_t(opts.topK), C.int32_t(opts.minTokens), C.int32_t(opts.threads),
		C.int(funcbool(opts.verbose)))
	expectSuccess(rc, "InitializeS2GenerateParams failed")
	return p
}

func initPipelineFromFiles(lib *s2, opts *options) unsafe.Pointer {
	p := requireHandle("AllocS2Pipeline", C.w_alloc(lib.allocPipeline))

	cModel := C.CString(opts.modelPath)
	cTok := C.CString(opts.tokenizerPath)
	defer C.free(unsafe.Pointer(cModel))
	defer C.free(unsafe.Pointer(cTok))

	codecFollow := C.int32_t(0)
	if opts.codecFollow {
		codecFollow = 1
	}

	rc := C.w_init_pipeline_files(lib.initPipeFiles, p,
		cModel, cTok,
		C.int32_t(opts.gpuDevice), C.int32_t(opts.backend),
		C.int32_t(opts.gpuLayers), codecFollow)
	expectSuccess(rc, "InitializeS2PipelineFromFiles failed")
	return p
}

func printSynthesizeResult(fn string, code C.int, sampleCount C.int32_t, outputPath string) {
	fmt.Printf("%s returned %d: %s\n", fn, int(code), describeSynthesizeCode(int(code)))
	fmt.Printf("samples=%d output=%s\n", int(sampleCount), orNone(outputPath))
}

func printStreamingResult(fn string, code C.int, s *streamingSession) {
	fmt.Printf("%s returned %d: %s\n", fn, int(code), describeStreamingCode(int(code)))
	errPart := ""
	if len(s.errors) > 0 {
		errPart = fmt.Sprintf(" errors=[%s]", strings.Join(s.errors, ", "))
	}
	fmt.Printf("chunks=%d bytes=%d output=%s done=%v%s\n",
		s.chunkCount, s.byteCount, orNone(s.outputPath), s.done, errPart)
}

func printAudioPreview(lib *s2, buffer unsafe.Pointer, sampleCount int) {
	if buffer == nil || sampleCount <= 0 {
		return
	}
	data := C.w_get_buffer_data(lib.getBufferData, buffer)
	if data == nil {
		return
	}
	n := 5
	if sampleCount < n {
		n = sampleCount
	}
	preview := make([]string, n)
	for i := 0; i < n; i++ {
		val := float32(C.s2_read_float(data, C.int32_t(i)))
		preview[i] = fmt.Sprintf("%.6f", val)
	}
	fmt.Printf("first_samples=[%s]\n", strings.Join(preview, ", "))
}

func isSynthesizeFailure(code int) bool {
	return code <= -4 || code == 0 || code == -6 || code == -7 || code == -8
}

func isStreamingFailure(code int) bool {
	return code <= -4 || code == 0 || code == -7 || code == -8 || code == -9 || code == -10
}

func describeSynthesizeCode(code int) string {
	switch {
	case code > 0:
		return "success"
	case code == 0:
		return "pipeline not initialized or invalid parameters"
	case code == -1:
		return "reference audio encode failed; synthesis continued without reference"
	case code == -4:
		return "synthesis failed or produced no frames"
	case code == -6:
		return "failed to save output audio"
	case code == -7:
		return "reference audio/codes require a non-empty transcript"
	case code == -8:
		return "precomputed prompt codes require ReferenceAudioTPrompt"
	default:
		return "unknown error"
	}
}

func describeStreamingCode(code int) string {
	switch {
	case code > 0:
		return "success"
	case code == 0:
		return "pipeline not initialized or invalid parameters"
	case code == -1:
		return "reference audio encode failed; synthesis continued without reference"
	case code == -4:
		return "streaming synthesis failed or produced no frames"
	case code == -7:
		return "reference audio/codes require a non-empty transcript"
	case code == -8:
		return "precomputed prompt codes require ReferenceAudioTPrompt"
	case code == -9:
		return "streaming callback configuration is invalid"
	case code == -10:
		return "streaming was aborted or cancelled by the client callback"
	default:
		return "unknown error"
	}
}

func funcbool(b bool) int {
	if b {
		return 1
	}
	return 0
}

func orNone(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}

func runSmoke(lib *s2) int {
	pipeline := requireHandle("AllocS2Pipeline", C.w_alloc(lib.allocPipeline))
	genParams := requireHandle("AllocS2GenerateParams", C.w_alloc(lib.allocGenParams))
	audioBuffer := requireHandle("AllocS2AudioBuffer", C.w_alloc_buffer(lib.allocBuffer, -1))
	promptCodes := requireHandle("AllocS2AudioPromptCodes", C.w_alloc(lib.allocCodes))
	model := requireHandle("AllocS2Model", C.w_alloc(lib.allocModel))
	tokenizer := requireHandle("AllocS2Tokenizer", C.w_alloc(lib.allocTokenizer))
	codec := requireHandle("AllocS2AudioCodec", C.w_alloc(lib.allocCodec))

	defer releaseHandle(codec, lib.releaseCodec)
	defer releaseHandle(tokenizer, lib.releaseTokenizer)
	defer releaseHandle(model, lib.releaseModel)
	defer releaseHandle(promptCodes, lib.releaseCodes)
	defer releaseHandle(audioBuffer, lib.releaseBuffer)
	defer releaseHandle(genParams, lib.releaseGenParams)
	defer releaseHandle(pipeline, lib.releasePipeline)

	rc := C.w_init_gen_params(lib.initGenParams, genParams, -1, -1, -1, -1, -1, -1, 0)
	expectSuccess(rc, "InitializeS2GenerateParams failed")

	logLevel := C.w_get_log_level(lib.getLogLevel)
	fmt.Printf("loaded library: log_level=%d\n", int(logLevel))
	fmt.Println("smoke test: exported symbols, allocators, params, and audio buffer OK")
	return 0
}

func runFromFiles(lib *s2, opts *options) int {
	opts.validateForSynthesis()

	pipeline := requireHandle("AllocS2Pipeline", C.w_alloc(lib.allocPipeline))
	genParams := requireHandle("AllocS2GenerateParams", C.w_alloc(lib.allocGenParams))
	audioBuffer := requireHandle("AllocS2AudioBuffer", C.w_alloc_buffer(lib.allocBuffer, -1))
	promptCodes := requireHandle("AllocS2AudioPromptCodes", C.w_alloc(lib.allocCodes))

	defer releaseHandle(promptCodes, lib.releaseCodes)
	defer releaseHandle(audioBuffer, lib.releaseBuffer)
	defer releaseHandle(genParams, lib.releaseGenParams)
	defer releaseHandle(pipeline, lib.releasePipeline)

	_ = initGenParams(lib, opts)
	_ = initPipelineFromFiles(lib, opts)

	var tPrompt C.int32_t
	var sampleCount C.int32_t

	cRef := cstring(opts.refAudioPath)
	cRefText := cstring(opts.refText)
	cText := cstring(opts.text)
	cOut := cstring(opts.outputPath)
	defer C.free(unsafe.Pointer(cRef))
	defer C.free(unsafe.Pointer(cRefText))
	defer C.free(unsafe.Pointer(cText))
	defer C.free(unsafe.Pointer(cOut))

	code := C.w_synthesize(lib.synthesize,
		pipeline, genParams, audioBuffer, promptCodes,
		&tPrompt, cRef, cRefText, cText, cOut, &sampleCount)

	printSynthesizeResult("S2Synthesize", code, sampleCount, opts.outputPath)
	printAudioPreview(lib, audioBuffer, int(sampleCount))

	if isSynthesizeFailure(int(code)) {
		return 1
	}
	return 0
}

func runModular(lib *s2, opts *options) int {
	opts.validateForSynthesis()

	pipeline := requireHandle("AllocS2Pipeline", C.w_alloc(lib.allocPipeline))
	genParams := requireHandle("AllocS2GenerateParams", C.w_alloc(lib.allocGenParams))
	audioBuffer := requireHandle("AllocS2AudioBuffer", C.w_alloc_buffer(lib.allocBuffer, -1))
	promptCodes := requireHandle("AllocS2AudioPromptCodes", C.w_alloc(lib.allocCodes))
	model := requireHandle("AllocS2Model", C.w_alloc(lib.allocModel))
	tokenizer := requireHandle("AllocS2Tokenizer", C.w_alloc(lib.allocTokenizer))
	codec := requireHandle("AllocS2AudioCodec", C.w_alloc(lib.allocCodec))

	defer releaseHandle(codec, lib.releaseCodec)
	defer releaseHandle(tokenizer, lib.releaseTokenizer)
	defer releaseHandle(model, lib.releaseModel)
	defer releaseHandle(promptCodes, lib.releaseCodes)
	defer releaseHandle(audioBuffer, lib.releaseBuffer)
	defer releaseHandle(genParams, lib.releaseGenParams)
	defer releaseHandle(pipeline, lib.releasePipeline)

	cTok := C.CString(opts.tokenizerPath)
	defer C.free(unsafe.Pointer(cTok))
	expectSuccess(C.w_init_tokenizer(lib.initTokenizer, tokenizer, cTok),
		"InitializeS2Tokenizer failed")

	cModel := C.CString(opts.modelPath)
	defer C.free(unsafe.Pointer(cModel))
	if opts.gpuLayers >= 0 {
		expectSuccess(C.w_init_model_gpu(lib.initModelGPU, model, cModel,
			C.int32_t(opts.gpuDevice), C.int32_t(opts.backend), C.int32_t(opts.gpuLayers)),
			"InitializeS2ModelWithGpuLayers failed")
	} else {
		expectSuccess(C.w_init_model(lib.initModel, model, cModel,
			C.int32_t(opts.gpuDevice), C.int32_t(opts.backend)),
			"InitializeS2Model failed")
	}

	codecGPU := C.int32_t(-1)
	codecBackend := C.int32_t(backendCPU)
	if opts.codecFollow {
		codecGPU = C.int32_t(opts.gpuDevice)
		codecBackend = C.int32_t(opts.backend)
	}
	expectSuccess(C.w_init_codec(lib.initCodec, codec, cModel, codecGPU, codecBackend),
		"InitializeS2AudioCodec failed")

	_ = initGenParams(lib, opts)

	C.w_sync_tokenizer(lib.syncTokenizer, model, tokenizer)
	expectSuccess(C.w_init_pipeline(lib.initPipeline, pipeline, tokenizer, model, codec),
		"InitializeS2Pipeline failed")

	var tPrompt C.int32_t
	if opts.refAudioPath != "" {
		cRef := C.CString(opts.refAudioPath)
		defer C.free(unsafe.Pointer(cRef))

		threads := opts.threads
		if threads <= 0 {
			threads = int32(runtime.NumCPU())
		}

		promptRC := C.w_init_prompt(lib.initCodes,
			pipeline, C.int32_t(threads), cRef, promptCodes, &tPrompt)
		if int(promptRC) == -1 {
			fmt.Println("InitializeAudioPromptCodes returned -1: reference encode failed, continuing without prompt codes")
			tPrompt = 0
		} else {
			expectSuccess(promptRC, "InitializeAudioPromptCodes failed")
		}
	}

	var sampleCount C.int32_t
	cText := cstring(opts.text)
	cRefText := cstring(opts.refText)
	cOut := cstring(opts.outputPath)
	defer C.free(unsafe.Pointer(cText))
	defer C.free(unsafe.Pointer(cRefText))
	defer C.free(unsafe.Pointer(cOut))

	code := C.w_synthesize(lib.synthesize,
		pipeline, genParams, audioBuffer, promptCodes,
		&tPrompt, nil, cRefText, cText, cOut, &sampleCount)

	printSynthesizeResult("S2Synthesize", code, sampleCount, opts.outputPath)
	printAudioPreview(lib, audioBuffer, int(sampleCount))

	if isSynthesizeFailure(int(code)) {
		return 1
	}
	return 0
}

func runLegacyStream(lib *s2, opts *options) int {
	opts.validateForSynthesis()

	genParams := initGenParams(lib, opts)
	pipeline := initPipelineFromFiles(lib, opts)
	promptCodes := requireHandle("AllocS2AudioPromptCodes", C.w_alloc(lib.allocCodes))

	defer releaseHandle(promptCodes, lib.releaseCodes)
	defer releaseHandle(genParams, lib.releaseGenParams)
	defer releaseHandle(pipeline, lib.releasePipeline)

	session := newStreamingSession(opts.outputPath, opts.play)
	userData := session.register()
	defer session.unregister()
	defer session.close()

	var cb C.struct_S2StreamingCallbacks
	C.build_streaming_callbacks(&cb, userData)

	var tPrompt C.int32_t
	cRef := cstring(opts.refAudioPath)
	cRefText := cstring(opts.refText)
	cText := cstring(opts.text)
	defer C.free(unsafe.Pointer(cRef))
	defer C.free(unsafe.Pointer(cRefText))
	defer C.free(unsafe.Pointer(cText))

	code := C.w_stream(lib.stream,
		pipeline, genParams, &cb, promptCodes,
		&tPrompt, cRef, cRefText, cText,
		C.int32_t(opts.streamStride))

	session.finalize()
	printStreamingResult("S2SynthesizeStreaming", code, session)

	if isStreamingFailure(int(code)) {
		return 1
	}
	return 0
}

func runStreamEx(lib *s2, opts *options) int {
	opts.validateForSynthesis()

	genParams := initGenParams(lib, opts)
	pipeline := initPipelineFromFiles(lib, opts)
	promptCodes := requireHandle("AllocS2AudioPromptCodes", C.w_alloc(lib.allocCodes))

	defer releaseHandle(promptCodes, lib.releaseCodes)
	defer releaseHandle(genParams, lib.releaseGenParams)
	defer releaseHandle(pipeline, lib.releasePipeline)

	session := newStreamingSession(opts.outputPath, opts.play)
	userData := session.register()
	defer session.unregister()
	defer session.close()

	var cb C.struct_S2StreamingCallbacks
	C.build_streaming_callbacks(&cb, userData)

	cVoice := cstring(opts.voice)
	cVoiceDir := cstring(opts.voiceDir)
	defer C.free(unsafe.Pointer(cVoice))
	defer C.free(unsafe.Pointer(cVoiceDir))

	sp := C.struct_S2StreamingParams{
		stream_decode_stride_frames: C.int32_t(opts.streamStride),
		stream_holdback_frames:      C.int32_t(opts.streamHoldback),
		codec_decode_context_frames: C.int32_t(opts.codecContext),
		low_latency:                 C.int(funcbool(opts.lowLatency)),
		segment_sentences:           C.int(funcbool(opts.segmentSentences)),
		sentence_pause_ms:           C.int32_t(opts.sentencePauseMs),
		segment_max_chars:           C.int32_t(opts.segmentMaxChars),
		voice:                       cVoice,
		voice_dir:                   cVoiceDir,
	}

	var tPrompt C.int32_t
	cRef := cstring(opts.refAudioPath)
	cRefText := cstring(opts.refText)
	cText := cstring(opts.text)
	defer C.free(unsafe.Pointer(cRef))
	defer C.free(unsafe.Pointer(cRefText))
	defer C.free(unsafe.Pointer(cText))

	code := C.w_stream_ex(lib.streamEx,
		pipeline, genParams, &cb, promptCodes,
		&tPrompt, cRef, cRefText, cText, &sp)

	session.finalize()
	printStreamingResult("S2SynthesizeStreamingEx", code, session)

	if isStreamingFailure(int(code)) {
		return 1
	}
	return 0
}

func main() {
	opts := parseArgs(os.Args[1:])

	lib, err := loadS2(opts.libraryPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer lib.close()

	C.w_set_log_level(lib.setLogLevel, C.int32_t(opts.logLevel))

	var code int
	switch opts.mode {
	case modeSmoke:
		code = runSmoke(lib)
	case modeFromFiles:
		code = runFromFiles(lib, opts)
	case modeModular:
		code = runModular(lib, opts)
	case modeLegacyStream:
		code = runLegacyStream(lib, opts)
	case modeStreamEx:
		code = runStreamEx(lib, opts)
	}
	os.Exit(code)
}
