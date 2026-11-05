// stego_core_cli <asset.json> <crop.pgm> [more crops (multi-frame)] [--reps N] [--H h0..h8]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "stego/stego.hpp"

static bool read_pgm(const char* path, std::vector<uint8_t>& px, int& w, int& h) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string magic;
  int maxv;
  f >> magic;
  auto skip = [&]() { while (f.peek() == '#' || f.peek() == '\n' || f.peek() == ' ' || f.peek() == '\r') { if (f.peek() == '#') { std::string l; std::getline(f, l); } else f.get(); } };
  skip(); f >> w; skip(); f >> h; skip(); f >> maxv; f.get();
  if (magic != "P5" || maxv != 255) return false;
  px.resize((size_t)w * h);
  f.read((char*)px.data(), px.size());
  return (bool)f;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s asset.json crop.pgm [crop2.pgm ...] [--reps N]\n", argv[0]); return 2; }
  std::vector<std::string> files;
  int reps = 1;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else files.push_back(argv[i]);
  }
  auto t0 = std::chrono::steady_clock::now();
  auto asset = stego::load_asset(argv[1]);
  double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::vector<std::vector<uint8_t>> px(files.size());
  std::vector<stego::ImageView> views(files.size());
  for (size_t i = 0; i < files.size(); i++) {
    int w, h;
    if (!read_pgm(files[i].c_str(), px[i], w, h)) { fprintf(stderr, "cannot read %s\n", files[i].c_str()); return 1; }
    views[i].data = px[i].data(); views[i].w = w; views[i].h = h;
  }
  printf("asset load %.1f ms\n", load_ms);
  for (int r = 0; r < reps; r++) {
    stego::Result R = views.size() == 1 ? stego::decode(*asset, views[0]) : stego::decode_multi(*asset, views.data(), (int)views.size());
    printf("rep %d: hex=%s crc=%d accept=%d z=%.2f amp=%.2fum n_valid=%d/%d reg_score=%.2f cand=%d\n", r, R.hex.c_str(), R.crc_ok, R.accept,
           R.z_pilot, R.amp_um, R.n_valid, R.n_points, R.reg_score, R.reg_cand);
    printf("  ms: dark %.2f coarse %.2f edge %.2f | nuis measure %.2f fit %.2f | tail %.2f golay %.2f | total %.2f\n", R.t.darkness, R.t.coarse,
           R.t.edge, R.t.measure, R.t.fit, R.t.tail, R.t.golay, R.t.total);
  }
  return 0;
}
