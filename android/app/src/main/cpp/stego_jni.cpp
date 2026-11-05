// JNI shim over core/ (stego::decode / decode_multi). Image data is read in place (pointer + row stride), never copied.
#include <jni.h>
#include <android/log.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <vector>
#include <stego/stego.hpp>
#include <stego/marks.hpp>
#include <stego/demo.hpp>

namespace {
std::vector<std::shared_ptr<const stego::Asset>> g_assets;   // index = handle returned by nInit
std::vector<std::unique_ptr<stego::MarkDetector>> g_dets;     // mark detectors (index = handle returned by nMarkInit)
const stego::Asset* asset_at(jint i) { return i >= 0 && (size_t)i < g_assets.size() ? g_assets[i].get() : nullptr; }

jobject make_result(JNIEnv* env, const stego::Result& r) {
  jclass cls = env->FindClass("me/stedra/stegocam/CoreResult");
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;ZZDDDI[DD[D[D[DDDD)V");
  jdoubleArray H = env->NewDoubleArray(9);
  env->SetDoubleArrayRegion(H, 0, 9, r.H);
  double t[8] = {r.t.darkness, r.t.coarse, r.t.edge, r.t.measure, r.t.fit, r.t.tail, r.t.golay, r.t.total};
  jdoubleArray T = env->NewDoubleArray(8);
  env->SetDoubleArrayRegion(T, 0, 8, t);
  jdoubleArray CY = env->NewDoubleArray((jsize)r.chip_y.size());
  env->SetDoubleArrayRegion(CY, 0, (jsize)r.chip_y.size(), r.chip_y.data());
  jdoubleArray CW = env->NewDoubleArray((jsize)r.chip_w.size());
  env->SetDoubleArrayRegion(CW, 0, (jsize)r.chip_w.size(), r.chip_w.data());
  return env->NewObject(cls, ctor, env->NewStringUTF(r.hex.c_str()), (jboolean)r.crc_ok, (jboolean)r.accept, (jdouble)r.z_pilot,
                        (jdouble)r.amp_um, (jdouble)r.amp_raw_um, (jint)r.n_valid, H, (jdouble)r.reg_score, T, CY, CW, (jdouble)r.pilot_s2, (jdouble)r.z_cal, (jdouble)r.null_sd);
}

// flags: bit1 = fast fitting, bit0 = wide registration windows (large blocks / bigger misregistration), same as the Python WIDE option set
stego::Params params_for(jint flags) {
  stego::Params p;
  if (flags & 2) { p.step = 0.02; p.iter_step = 0.04; p.fit_stride = 4; p.iter_stride = 8; }   // fast: coarser taps / sparser fit (z unchanged on 128-bit block captures, ~35% faster)
  if (flags & 1) { p.reg_W = {2.4, 1.2, 0.6, 0.3}; p.reg_iters = {4, 3, 3, 2}; p.reg_step = {0.2, 0.1, 0.05, 0.03}; }
  return p;
}

stego::ImageView view(const void* p, jint w, jint h, jint stride) {
  stego::ImageView v;
  v.data = p; v.w = w; v.h = h; v.stride = stride; v.is_float = false;
  return v;
}
}  // namespace

extern "C" {

JNIEXPORT jint JNICALL Java_me_stedra_stegocam_StegoCore_nInit(JNIEnv* env, jclass, jstring json, jbyteArray bin) {
  try {
    const char* js = env->GetStringUTFChars(json, nullptr);
    std::string text(js);
    env->ReleaseStringUTFChars(json, js);
    jsize n = env->GetArrayLength(bin);
    std::vector<uint8_t> b(n);
    env->GetByteArrayRegion(bin, 0, n, (jbyte*)b.data());
    g_assets.push_back(stego::load_asset_mem(text, b.data(), b.size()));
    return (jint)g_assets.size() - 1;
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "load_asset failed: %s", e.what());
    return -1;
  }
}

// addr: address of the first pixel (uint8, e.g. Mat.dataAddr() of a crop, or a direct ByteBuffer address); stride in bytes.
JNIEXPORT jobject JNICALL Java_me_stedra_stegocam_StegoCore_nDecode(JNIEnv* env, jclass, jint ai, jlong addr, jint w, jint h, jint stride, jint flags) {
  const stego::Asset* A = asset_at(ai);
  if (!A) return nullptr;
  try {
    stego::Result r = stego::decode(*A, view((const void*)addr, w, h, stride), params_for(flags));
    return make_result(env, r);
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "decode failed: %s", e.what());
    return nullptr;
  }
}

// direct ByteBuffer variant (position offset in bytes), e.g. an ImageReader plane
JNIEXPORT jobject JNICALL Java_me_stedra_stegocam_StegoCore_nDecodeBuf(JNIEnv* env, jclass, jint ai, jobject buf, jint off, jint w, jint h, jint stride, jint flags) {
  const stego::Asset* A = asset_at(ai);
  if (!A) return nullptr;
  uint8_t* p = (uint8_t*)env->GetDirectBufferAddress(buf);
  if (!p) return nullptr;
  try {
    stego::Result r = stego::decode(*A, view(p + off, w, h, stride), params_for(flags));
    return make_result(env, r);
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "decode failed: %s", e.what());
    return nullptr;
  }
}

JNIEXPORT jobject JNICALL Java_me_stedra_stegocam_StegoCore_nDecodeMulti(JNIEnv* env, jclass, jint ai, jlongArray addrs, jintArray ws, jintArray hs, jintArray strides, jint flags) {
  const stego::Asset* A = asset_at(ai);
  if (!A) return nullptr;
  jsize k = env->GetArrayLength(addrs);
  std::vector<jlong> a(k); std::vector<jint> w(k), h(k), s(k);
  env->GetLongArrayRegion(addrs, 0, k, a.data());
  env->GetIntArrayRegion(ws, 0, k, w.data());
  env->GetIntArrayRegion(hs, 0, k, h.data());
  env->GetIntArrayRegion(strides, 0, k, s.data());
  std::vector<stego::ImageView> v(k);
  for (int i = 0; i < k; i++) v[i] = view((const void*)a[i], w[i], h[i], s[i]);
  try {
    stego::Result r = stego::decode_multi(*A, v.data(), k, params_for(flags));
    return make_result(env, r);
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "decode_multi failed: %s", e.what());
    return nullptr;
  }
}

// decode from accumulated chip statistics (see stego::finalize_chips)
JNIEXPORT jobject JNICALL Java_me_stedra_stegocam_StegoCore_nFinalize(JNIEnv* env, jclass, jint ai, jdoubleArray y, jdoubleArray w) {
  const stego::Asset* A = asset_at(ai);
  if (!A) return nullptr;
  jsize n = env->GetArrayLength(y);
  std::vector<double> yy(n), ww(n);
  env->GetDoubleArrayRegion(y, 0, n, yy.data());
  env->GetDoubleArrayRegion(w, 0, n, ww.data());
  try {
    return make_result(env, stego::finalize_chips(*A, yy.data(), ww.data()));
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "finalize failed: %s", e.what());
    return nullptr;
  }
}

// ---- marks: detector built from an already loaded asset (geometry only), detect + decode in one call
JNIEXPORT jint JNICALL Java_me_stedra_stegocam_StegoCore_nMarkInit(JNIEnv*, jclass, jint ai) {
  const stego::Asset* A = asset_at(ai);
  if (!A) return -1;
  try {
    g_dets.push_back(std::make_unique<stego::MarkDetector>(*A));
    return (jint)g_dets.size() - 1;
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "mark detector init failed: %s", e.what());
    return -1;
  }
}

// flags: bit0 wide windows, bit1 fast fit (as nDecode), bit2 skip the refine retry, bit3 detect only (no decode)
// exclude: triples (cx, cy, r) in view px of already handled marks
JNIEXPORT jobject JNICALL Java_me_stedra_stegocam_StegoCore_nMarkDecode(JNIEnv* env, jclass, jint di, jintArray assets, jlong addr, jint w, jint h, jint stride,
                                                                        jint flags, jdouble ppmLo, jdouble ppmHi, jint sym, jdoubleArray exclude) {
  if (di < 0 || (size_t)di >= g_dets.size()) return nullptr;
  std::vector<const stego::Asset*> as;
  {
    jsize n = env->GetArrayLength(assets);
    std::vector<jint> ids(n);
    env->GetIntArrayRegion(assets, 0, n, ids.data());
    for (jint id : ids) { const stego::Asset* a = asset_at(id); if (a) as.push_back(a); }
  }
  stego::MarkDecodeOpts o;
  o.dec = params_for(flags);
  o.det.ppm_lo = ppmLo; o.det.ppm_hi = ppmHi; o.det.sym = sym > 0 ? sym : 1;
  o.refine = !(flags & 4);
  if (exclude) {
    jsize n = env->GetArrayLength(exclude);
    std::vector<jdouble> ex(n);
    env->GetDoubleArrayRegion(exclude, 0, n, ex.data());
    for (jsize i = 0; i + 2 < n; i += 3) o.det.exclude.push_back({ex[i], ex[i + 1], ex[i + 2]});
  }
  try {
    stego::ImageView v = view((const void*)addr, w, h, stride);
    stego::MarkDecodeResult r;
    if (flags & 8) { r.det = g_dets[di]->detect(v, o.det); r.ms_total = r.det.t.total; }
    else r = stego::detect_and_decode(*g_dets[di], as, v, o);
    jclass cls = env->FindClass("me/stedra/stegocam/MarkOutcome");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ZDDDDDD[DIIZ[IDDDI[DLme/stedra/stegocam/CoreResult;)V");
    jdoubleArray H = env->NewDoubleArray(9);
    env->SetDoubleArrayRegion(H, 0, 9, r.det.H);
    jintArray crop = env->NewIntArray(4);
    env->SetIntArrayRegion(crop, 0, 4, r.crop);
    double dt[6] = {r.det.t.prep, r.det.t.stage_a, r.det.t.polar, r.det.t.fine, r.det.t.ecc, r.det.t.total};
    jdoubleArray D = env->NewDoubleArray(6);
    env->SetDoubleArrayRegion(D, 0, 6, dt);
    jobject core = r.asset >= 0 ? make_result(env, r.dec) : nullptr;
    return env->NewObject(cls, ctor, (jboolean)r.det.found, (jdouble)r.det.score, (jdouble)r.det.iso, (jdouble)r.det.ppm, (jdouble)r.det.rot_deg,
                          (jdouble)r.det.centre[0], (jdouble)r.det.centre[1], H, (jint)r.asset, (jint)r.hyp, (jboolean)r.refined, crop,
                          (jdouble)r.det.t.total, (jdouble)r.ms_decode, (jdouble)r.ms_total, (jint)r.n_decodes, D, core);
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "mark decode failed: %s", e.what());
    return nullptr;
  }
}

// ---- demo cards: seed H (comp page mm -> frame px) -> edge registration of the whole design, 3 regions, combined token
// flags: bit1 fast fit. shifts = 2 doubles per region.
JNIEXPORT jobject JNICALL Java_me_stedra_stegocam_StegoCore_nDemoDecode(JNIEnv* env, jclass, jint comp, jintArray regions, jdoubleArray shifts, jlong addr, jint w, jint h,
                                                                        jint stride, jdoubleArray Hseed, jint flags) {
  const stego::Asset* C = asset_at(comp);
  if (!C) return nullptr;
  std::vector<const stego::Asset*> rs;
  {
    jsize n = env->GetArrayLength(regions);
    std::vector<jint> ids(n);
    env->GetIntArrayRegion(regions, 0, n, ids.data());
    for (jint id : ids) { const stego::Asset* a = asset_at(id); if (!a) return nullptr; rs.push_back(a); }
  }
  std::vector<std::array<double, 2>> sh(rs.size());
  {
    std::vector<jdouble> v(2 * rs.size());
    env->GetDoubleArrayRegion(shifts, 0, (jsize)v.size(), v.data());
    for (size_t i = 0; i < rs.size(); i++) sh[i] = {v[2 * i], v[2 * i + 1]};
  }
  double H[9];
  env->GetDoubleArrayRegion(Hseed, 0, 9, H);
  try {
    stego::DemoDecodeResult r = stego::demo_decode(*C, rs, sh, view((const void*)addr, w, h, stride), H, (flags & 2) != 0);
    jclass cls = env->FindClass("me/stedra/stegocam/DemoOutcome");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ZZZLjava/lang/String;D[D[DDZ[D[IDDD)V");
    jdoubleArray zs = env->NewDoubleArray((jsize)r.comb.zs.size());
    env->SetDoubleArrayRegion(zs, 0, (jsize)r.comb.zs.size(), r.comb.zs.data());
    jdoubleArray gains = env->NewDoubleArray((jsize)r.comb.gains.size());
    env->SetDoubleArrayRegion(gains, 0, (jsize)r.comb.gains.size(), r.comb.gains.data());
    jdoubleArray HH = env->NewDoubleArray(9);
    env->SetDoubleArrayRegion(HH, 0, 9, r.H);
    jintArray crop = env->NewIntArray(4);
    env->SetIntArrayRegion(crop, 0, 4, r.crop);
    return env->NewObject(cls, ctor, (jboolean)r.comb.accept, (jboolean)r.comb.crc_ok, (jboolean)r.comb.tag_ok, env->NewStringUTF(r.comb.token.c_str()), (jdouble)r.comb.z,
                          zs, gains, (jdouble)r.reg_score, (jboolean)r.wide, HH, crop, (jdouble)r.ms_reg, (jdouble)r.ms_dec, (jdouble)r.ms_total);
  } catch (const std::exception& e) {
    __android_log_print(ANDROID_LOG_ERROR, "StegoCam", "demo decode failed: %s", e.what());
    return nullptr;
  }
}

// Pin thread tid to the CPUs in mask (bit i = cpu i). Returns 0, or errno. mask 0 = all cpus.
JNIEXPORT jint JNICALL Java_me_stedra_stegocam_StegoCore_nSetAffinity(JNIEnv*, jclass, jint tid, jlong mask) {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (mask == 0) { long n = sysconf(_SC_NPROCESSORS_CONF); for (long i = 0; i < n && i < CPU_SETSIZE; i++) CPU_SET(i, &set); }
  else for (int i = 0; i < 64; i++) if ((mask >> i) & 1) CPU_SET(i, &set);
  return sched_setaffinity(tid, sizeof set, &set) == 0 ? 0 : errno;
}

JNIEXPORT jlong JNICALL Java_me_stedra_stegocam_StegoCore_nGetAffinity(JNIEnv*, jclass, jint tid) {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(tid, sizeof set, &set) != 0) return -1;
  jlong m = 0;
  for (int i = 0; i < 64; i++) if (CPU_ISSET(i, &set)) m |= 1LL << i;
  return m;
}

}
