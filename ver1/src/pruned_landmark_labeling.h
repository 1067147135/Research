
#ifndef PRUNED_LANDMARK_LABELING_H_
#define PRUNED_LANDMARK_LABELING_H_

#include <malloc.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <sys/time.h>
#include <climits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <set>
#include <algorithm>
#include <fstream>
#include <utility>
#include "omp.h"

#define NumThreads 8
#define constrain 7

class PrunedLandmarkLabeling {
 public:
  // Constructs an index from a graph, given as a list of edges.
  // Vertices should be described by numbers starting from zero.
  // Returns |true| when successful.
  bool ConstructIndex(const std::vector<std::pair<int, int> > &es);
  bool ConstructIndex(std::istream &ifs);
  bool ConstructIndex(const char *filename);

  // Returns distance vetween vertices |v| and |w| if they are connected.
  // Otherwise, returns |INT_MAX|.
  inline int QueryDistance(int v, int w);

  // Loads an index. Returns |true| when successful.
  bool LoadIndex(std::istream &ifs);
  bool LoadIndex(const char *filename);

  // Stores the index. Returns |true| when successful.
  bool StoreIndex(std::ostream &ofs);
  bool StoreIndex(const char *filename);

  int GetNumVertices() { return num_v_; }
  void Free();
  void PrintStatistics();

  // bfs check
  inline int DistanceCheck(int s, int t);

  //dfs test
  int dfs(int s, int t, int step, int ele);
  int parallel_dfs(int s, int t, int step, int ele);
  int para_dfs(int s, int t, int step, int ele, long id);

  PrunedLandmarkLabeling()
      : adj(NULL), index_in_(NULL), index_out_(NULL), time_load_(0), time_indexing_(0) {}
  virtual ~PrunedLandmarkLabeling() {
    Free();
  }

 private:
  static const uint8_t INF8;  // For unreachable pairs

  struct index_t {
    uint32_t *spt_v;
    uint8_t *spt_d;
  } __attribute__((aligned(64)));  // Aligned for cache lines

  
  
  // bfs check
  struct neighbor{
    uint32_t *nb;
  } __attribute__((aligned(64))); ;
  
  neighbor *adj;

  index_t *index_in_;
  index_t *index_out_;

  double GetCurrentTimeSec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

  // Statistics
  double time_load_, time_indexing_;

public:
  // dfs test
  int num_v_;
  int stack[constrain + 1]; // k = 7
  bool* visited;
  int count;
  int count_sum[NumThreads];
};

const uint8_t PrunedLandmarkLabeling::INF8 = 100;

bool PrunedLandmarkLabeling
::ConstructIndex(const char *filename) {
  std::ifstream ifs(filename);
  // std::cout << "filename: " << filename << std::endl;
  return ifs && ConstructIndex(ifs);
}

bool PrunedLandmarkLabeling
::ConstructIndex(std::istream &ifs) { // only use the part "load graph"
  std::vector<std::pair<int, int> > es;
  for (int v, w; ifs >> v >> w; ) {
    // std::cout << "v: " << v << " w: " << w << std::endl;
    es.push_back(std::make_pair(v, w));
  }
  if (ifs.bad()) return false;
  std::cout << es.size() << std::endl;
  Free();
  time_load_ = -GetCurrentTimeSec();
  int &V = num_v_;  // number of vertices, count from 0
  V = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    V = std::max(V, std::max(es[i].first, es[i].second) + 1);
  }// V = the max tag of vertex + 1.  
  std::vector<std::vector<int> > adj_in(V);
  std::vector<std::vector<int> > adj_out(V);
  for (size_t i = 0; i < es.size(); ++i) {
    int v = es[i].first, w = es[i].second;
    adj_in[w].push_back(v);
    adj_out[v].push_back(w);  
  }
  time_load_ += GetCurrentTimeSec();

  // copy original adj_out
  adj = (neighbor*)memalign(64, V * sizeof(neighbor));
  visited = (bool*)memalign(64, V * sizeof(bool));
  for (int v = 0; v < V; ++v) { 
    int k = adj_out[v].size();
    adj[v].nb = (uint32_t*)memalign(64, (k + 1) * sizeof(uint32_t));
    for (int j = 0; j < k; j++){
      adj[v].nb[j] = adj_out[v][j];
    }
    adj[v].nb[k] = V + 1;
    visited[v] = false;
  }

  std::cout << "num_v_: " << num_v_ << " V: " << V << std::endl;
  // ConstructIndex(es);
  return true;
}

bool PrunedLandmarkLabeling
::ConstructIndex(const std::vector<std::pair<int, int> > &es) {
  //
  // Prepare the adjacency list and index space
  //
  Free();
  time_load_ = -GetCurrentTimeSec();
  int &V = num_v_;  // number of vertices, count from 0
  V = 0;
  for (size_t i = 0; i < es.size(); ++i) {
    V = std::max(V, std::max(es[i].first, es[i].second) + 1);
  }// V = the max tag of vertex + 1.  
  std::vector<std::vector<int> > adj_in(V);
  std::vector<std::vector<int> > adj_out(V);
  for (size_t i = 0; i < es.size(); ++i) {
    int v = es[i].first, w = es[i].second;
    adj_in[w].push_back(v);
    adj_out[v].push_back(w);  
  }
  time_load_ += GetCurrentTimeSec();

  // copy original adj_out
  adj = (neighbor*)memalign(64, V * sizeof(neighbor));
  for (int v = 0; v < V; ++v) { 
    int k = adj_out[v].size();
    adj[v].nb = (uint32_t*)memalign(64, (k + 1) * sizeof(uint32_t));
    for (int j = 0; j < k; j++){
      adj[v].nb[j] = adj_out[v][j];
    }
    adj[v].nb[k] = V + 1;
  }

  // void * memalign (size_t boundary, size_t size) 
  index_in_ = (index_t*)memalign(64, V * sizeof(index_t));
  index_out_ = (index_t*)memalign(64, V * sizeof(index_t));

  if (index_in_ == NULL || index_out_ == NULL) { 
    num_v_ = 0;
    return false;
  }
  for (int v = 0; v < V; ++v) { 
    index_in_[v].spt_v = NULL;
    index_in_[v].spt_d = NULL;
    index_out_[v].spt_v = NULL;
    index_out_[v].spt_d = NULL;
  }

  //
  // Order vertices by decreasing order of degree
  //
  time_indexing_ = -GetCurrentTimeSec();
  std::vector<int> inv(V);  
  {
    // Order
    std::vector<std::pair<float, int> > deg(V);
    for (int v = 0; v < V; ++v) {
      // We add a random value here to diffuse nearby vertices
      // + float(rand()) / RAND_MAX
      deg[v] = std::make_pair(adj_out[v].size(), v);
    }

    std::sort(deg.rbegin(), deg.rend());
    for (int i = 0; i < V; ++i) inv[i] = deg[i].second; // inv[new label] = old label
    

    // Relabel the vertex IDs
    std::vector<int> rank(V); // rank[old label] = new label
    
    for (int i = 0; i < V; ++i){
      rank[deg[i].second] = i;
    } 
    
    std::vector<std::vector<int> > new_adj_in(V);
    std::vector<std::vector<int> > new_adj_out(V);
    for (int v = 0; v < V; ++v) {
      for (size_t i = 0; i < adj_in[v].size(); ++i) {  
        new_adj_in[rank[v]].push_back(rank[adj_in[v][i]]);
      }
      for (size_t i = 0; i < adj_out[v].size(); ++i) {  
        new_adj_out[rank[v]].push_back(rank[adj_out[v][i]]);
      }
    }
    adj_in.swap(new_adj_in); 
    adj_out.swap(new_adj_out);
  }

  //
  // Bit-parallel labeling
  //
  // ??????V??0??false??
  std::vector<bool> usd(V, false);  // Used as root? (in new label)
  //
  // Pruned labeling
  //
  // pruned BFSs using normal labels for pruning
  {
    // Sentinel (V, INF8) is added to all the vertices
    std::vector<std::vector<std::pair<int, uint8_t>>>
        tmp_idx_in(V, (std::vector<std::pair<int, uint8_t>>(1, std::make_pair(V, INF8))));
    std::vector<std::vector<std::pair<int, uint8_t>>>
        tmp_idx_out(V, (std::vector<std::pair<int, uint8_t>>(1, std::make_pair(V, INF8))));

    std::vector<bool> vis_in(V);
    std::vector<bool> vis_out(V);
    std::vector<int> que_in(V);  // queue
    std::vector<int> que_out(V);  // queue
    std::vector<uint8_t> dst_r_in(V + 1, INF8);  // distance to r
    std::vector<uint8_t> dst_r_out(V + 1, INF8);  // distance to r

    for (int r = 0; r < V; ++r) { 
      //if (usd[r]) continue; 
      std::vector<std::pair<int, uint8_t>>
          &tmp_idx_r_in = tmp_idx_in[r];  
      std::vector<std::pair<int, uint8_t>>
          &tmp_idx_r_out = tmp_idx_out[r];  
      for (size_t i = 0; i < tmp_idx_r_in.size(); ++i) {
        dst_r_in[tmp_idx_r_in[i].first] = tmp_idx_r_in[i].second;
      }
      for (size_t i = 0; i < tmp_idx_r_out.size(); ++i) {
        dst_r_out[tmp_idx_r_out[i].first] = tmp_idx_r_out[i].second;
      }

      int que_t0_in = 0, que_t1_in = 0, que_h_in = 0;
      int que_t0_out = 0, que_t1_out = 0, que_h_out = 0;  // que_t0 ~ que_t1: ????????????que_t1 ~ que_h: ????????????????
      que_in[que_h_in++] = r;
      que_out[que_h_out++] = r;
      vis_in[r] = true;
      vis_out[r] = true;
      que_t1_in = que_h_in;
      que_t1_out = que_h_out;

      for (uint8_t d = 0; que_t0_in < que_h_in || que_t0_out < que_h_out; ++d) {
        if (que_t0_in < que_h_in){
          for (int que_i_in = que_t0_in; que_i_in < que_t1_in; ++que_i_in) {
            int v = que_in[que_i_in];
            std::vector<std::pair<int, uint8_t>>
                &tmp_idx_v_out = tmp_idx_out[v];

            // Prefetch
            _mm_prefetch(&tmp_idx_v_out[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_r_in[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_v_out[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_r_in[0], _MM_HINT_T0);

            //Prune?
            if (usd[v]){
              // tmp_idx_v_in.back() = std::make_pair(r, d);  
              // tmp_idx_r_out.back() = std::make_pair(v, d);
              // tmp_idx_v_in.push_back(std::make_pair(V, INF8));
              // tmp_idx_r_out.push_back(std::make_pair(V, INF8));
              // dst_r[v] = d;
              continue;
            } 
            

            for (size_t i = 0; i < tmp_idx_v_out.size(); ++i) {
              int w = tmp_idx_v_out[i].first;
              int td = tmp_idx_v_out[i].second + dst_r_in[w];
              if (td <= d) goto pruned_in;
            }
            
            // Traverse
            tmp_idx_v_out.back() = std::make_pair(r, d);  
            tmp_idx_r_in.back() = std::make_pair(v, d);
            tmp_idx_v_out.push_back(std::make_pair(V, INF8));
            tmp_idx_r_in.push_back(std::make_pair(V, INF8));
            dst_r_in[v] = d;


            for (size_t i = 0; i < adj_in[v].size(); ++i) {
              int w = adj_in[v][i];
              if (!vis_in[w]) {
                que_in[que_h_in++] = w;
                vis_in[w] = true;
              }
            }
          pruned_in:
            {}
          }

          que_t0_in = que_t1_in;
          que_t1_in = que_h_in;
        }
        if (que_t0_out < que_h_out){
          for (int que_i_out = que_t0_out; que_i_out < que_t1_out; ++que_i_out) {
            int v = que_out[que_i_out];
            std::vector<std::pair<int, uint8_t>>
                &tmp_idx_v_in = tmp_idx_in[v];

            // Prefetch
            _mm_prefetch(&tmp_idx_v_in[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_r_out[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_v_in[0], _MM_HINT_T0);
            _mm_prefetch(&tmp_idx_r_out[0], _MM_HINT_T0);

            // Prune?
            if (usd[v]){
              // tmp_idx_v_in.back() = std::make_pair(r, d);  
              // tmp_idx_r_out.back() = std::make_pair(v, d);
              // tmp_idx_v_in.push_back(std::make_pair(V, INF8));
              // tmp_idx_r_out.push_back(std::make_pair(V, INF8));
              // dst_r[v] = d;
              continue;
            } 
            

            for (size_t i = 0; i < tmp_idx_v_in.size(); ++i) {
              int w = tmp_idx_v_in[i].first;
              int td = tmp_idx_v_in[i].second + dst_r_out[w];
              if (td <= d) goto pruned_out;
            }
            
            // Traverse
            tmp_idx_v_in.back() = std::make_pair(r, d);  
            tmp_idx_r_out.back() = std::make_pair(v, d);
            tmp_idx_v_in.push_back(std::make_pair(V, INF8));
            tmp_idx_r_out.push_back(std::make_pair(V, INF8));
            dst_r_out[v] = d;


            for (size_t i = 0; i < adj_out[v].size(); ++i) {
              int w = adj_out[v][i];
              if (!vis_out[w]) {
                que_out[que_h_out++] = w;
                vis_out[w] = true;
              }
            }
          pruned_out:
            {}
          }

          que_t0_out = que_t1_out;
          que_t1_out = que_h_out;
        }
        
      }

      for (int i = 0; i < que_h_in; ++i) vis_in[que_in[i]] = false;  
      for (int i = 0; i < que_h_out; ++i) vis_out[que_out[i]] = false;  
      for (size_t i = 0; i < tmp_idx_r_in.size(); ++i) {
        dst_r_in[tmp_idx_r_in[i].first] = INF8;
      }
      for (size_t i = 0; i < tmp_idx_r_out.size(); ++i) {
        dst_r_out[tmp_idx_r_out[i].first] = INF8;
      }
      usd[r] = true;
    }

    // std::cout << "tmp_idx_in:" << std::endl;
    // for (size_t j = 0; j < tmp_idx_in.size(); j++){
    //   for (size_t i = 0; i < tmp_idx_in[j].size(); i++){
    //     std::cout << inv[j] << ":   " << inv[tmp_idx_in[j][i].first] << "  " << unsigned(tmp_idx_in[j][i].second) << std::endl;
    //   }
    // }
    // std::cout << "tmp_idx_out:" << std::endl;
    // for (size_t j = 0; j < tmp_idx_out.size(); j++){
    //   for (size_t i = 0; i < tmp_idx_out[j].size(); i++){
    //     std::cout << inv[j] << ":   " << inv[tmp_idx_out[j][i].first] << "  " << unsigned(tmp_idx_out[j][i].second) << std::endl;
    //   }
    // }

    for (int v = 0; v < V; ++v) {
      int k1 = tmp_idx_in[v].size();
      index_in_[inv[v]].spt_v = (uint32_t*)memalign(64, k1 * sizeof(uint32_t));
      index_in_[inv[v]].spt_d = (uint8_t *)memalign(64, k1 * sizeof(uint8_t ));
      int k2 = tmp_idx_out[v].size();
      index_out_[inv[v]].spt_v = (uint32_t*)memalign(64, k2 * sizeof(uint32_t));
      index_out_[inv[v]].spt_d = (uint8_t *)memalign(64, k2 * sizeof(uint8_t ));
      if (!index_in_[inv[v]].spt_v || !index_in_[inv[v]].spt_d || !index_out_[inv[v]].spt_v || !index_out_[inv[v]].spt_d) {
        Free();
        return false;
      }
      sort(tmp_idx_in[v].begin(), tmp_idx_in[v].end());
      for (int i = 0; i < k1; ++i){
        index_in_[inv[v]].spt_v[i] = tmp_idx_in[v][i].first;
        index_in_[inv[v]].spt_d[i] = tmp_idx_in[v][i].second;
      }
        
      sort(tmp_idx_out[v].begin(), tmp_idx_out[v].end());
      for (int i = 0; i < k2; ++i) {
        index_out_[inv[v]].spt_v[i] = tmp_idx_out[v][i].first;
        index_out_[inv[v]].spt_d[i] = tmp_idx_out[v][i].second;
      }

      

      tmp_idx_in[v].clear();
      tmp_idx_in[v].clear();
      tmp_idx_out[v].clear();
      tmp_idx_out[v].clear();


    }
  }

  time_indexing_ += GetCurrentTimeSec();    
  
  return true;
}


int PrunedLandmarkLabeling
::QueryDistance(int v, int w) {
  if (v >= num_v_ || w >= num_v_) return v == w ? 0 : INT_MAX;  // INT_MAX = 2147483647

  const index_t &idx_v = index_out_[v];
  const index_t &idx_w = index_in_[w];
  int d = INF8;

  _mm_prefetch(&idx_v.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_v[0], _MM_HINT_T0);
  _mm_prefetch(&idx_v.spt_d[0], _MM_HINT_T0);
  _mm_prefetch(&idx_w.spt_d[0], _MM_HINT_T0);

  // std::cout << "query solving:" << std::endl;
  for (int i1 = 0, i2 = 0; ; ) {
    int v1 = idx_v.spt_v[i1], v2 = idx_w.spt_v[i2];
    // std::cout << v1 << "  " << v2 << std::endl;
    if (v1 == v2) {
      if (v1 == num_v_) break;  // Sentinel
      int td = idx_v.spt_d[i1] + idx_w.spt_d[i2];
      if (td < d) d = td;
      ++i1;
      ++i2;
    } else {
      i1 += v1 < v2 ? 1 : 0;
      i2 += v1 > v2 ? 1 : 0;
    }
  }

  if (d >= INF8 - 2) d = INT_MAX;
  return d;
}


bool PrunedLandmarkLabeling
::LoadIndex(const char *filename) {
  std::ifstream ifs(filename);
  return ifs && LoadIndex(ifs);
}

bool PrunedLandmarkLabeling
::LoadIndex(std::istream &ifs) {
  Free();

  int32_t num_v;
  ifs.read((char*)&num_v,   sizeof(num_v));

  num_v_ = num_v;
  if (ifs.bad()) {
    num_v_ = 0;
    return false;
  }

  index_in_ = (index_t*)memalign(64, num_v * sizeof(index_t));
  index_out_ = (index_t*)memalign(64, num_v * sizeof(index_t));
  if (index_in_ == NULL || index_out_ == NULL) {
    num_v_ = 0;
    return false;
  }
  for (int v = 0; v < num_v_; ++v) {
    index_in_[v].spt_v = NULL;
    index_in_[v].spt_d = NULL;
    index_out_[v].spt_v = NULL;
    index_out_[v].spt_d = NULL;
  }

  for (int v = 0; v < num_v_; ++v) {
    index_t &idx_in = index_in_[v];
    index_t &idx_out = index_out_[v];

    int32_t s1;
    ifs.read((char*)&s1, sizeof(s1));
    if (ifs.bad()) {
      Free();
      return false;
    }

    idx_in.spt_v = (uint32_t*)memalign(64, s1 * sizeof(uint32_t));
    idx_in.spt_d = (uint8_t *)memalign(64, s1 * sizeof(uint8_t ));
    if (!idx_in.spt_v || !idx_in.spt_d) {
      Free();
      return false;
    }

    for (int i = 0; i < s1; ++i) {
      ifs.read((char*)&idx_in.spt_v[i], sizeof(idx_in.spt_v[i]));
      ifs.read((char*)&idx_in.spt_d[i], sizeof(idx_in.spt_d[i]));
    }

    int32_t s2;
    ifs.read((char*)&s2, sizeof(s2));
    if (ifs.bad()) {
      Free();
      return false;
    }

    idx_out.spt_v = (uint32_t*)memalign(64, s2 * sizeof(uint32_t));
    idx_out.spt_d = (uint8_t *)memalign(64, s2 * sizeof(uint8_t ));
    if (!idx_out.spt_v || !idx_out.spt_d) {
      Free();
      return false;
    }

    for (int i = 0; i < s2; ++i) {
      ifs.read((char*)&idx_out.spt_v[i], sizeof(idx_out.spt_v[i]));
      ifs.read((char*)&idx_out.spt_d[i], sizeof(idx_out.spt_d[i]));
    }
  }

  return ifs.good();
}


bool PrunedLandmarkLabeling
::StoreIndex(const char *filename) {
  std::ofstream ofs(filename);
  return ofs && StoreIndex(ofs);
}


bool PrunedLandmarkLabeling
::StoreIndex(std::ostream &ofs) {
  uint32_t num_v = num_v_;
  ofs.write((const char*)&num_v,   sizeof(num_v));

  for (int v = 0; v < num_v_; ++v) {
    index_t &idx_in = index_in_[v];
    index_t &idx_out = index_out_[v];

    int32_t s1;
    for (s1 = 1; idx_in.spt_v[s1 - 1] != num_v; ++s1) continue;  // Find the sentinel
    ofs.write((const char*)&s1, sizeof(s1));
    for (int i = 0; i < s1; ++i) {
      int32_t l = idx_in.spt_v[i];
      int8_t  d = idx_in.spt_d[i];
      ofs.write((const char*)&l, sizeof(l));
      ofs.write((const char*)&d, sizeof(d));
    }
    int32_t s2;
    for (s2 = 1; idx_out.spt_v[s2 - 1] != num_v; ++s2) continue;  // Find the sentinel
    ofs.write((const char*)&s2, sizeof(s2));
    for (int i = 0; i < s2; ++i) {
      int32_t l = idx_out.spt_v[i];
      int8_t  d = idx_out.spt_d[i];
      ofs.write((const char*)&l, sizeof(l));
      ofs.write((const char*)&d, sizeof(d));
    }
  }

  return ofs.good();
}


void PrunedLandmarkLabeling
::Free() {
  for (int v = 0; v < num_v_; ++v) {
    free(index_in_[v].spt_v);
    free(index_in_[v].spt_d);
    free(index_out_[v].spt_v);
    free(index_out_[v].spt_d);
  }
  free(index_in_);
  free(index_out_);
  index_in_ = NULL;
  index_out_ = NULL;
  num_v_ = 0;
}

void PrunedLandmarkLabeling
::PrintStatistics() {
  std::cout << "load time: "     << time_load_     << " seconds" << std::endl;
  std::cout << "indexing time: " << time_indexing_ << " seconds" << std::endl;

  double s = 0.0;
  for (int v = 0; v < num_v_; ++v) {
    for (int i = 0; index_in_[v].spt_v[i] != uint32_t(num_v_); ++i) {
      ++s;
    }
    for (int i = 0; index_out_[v].spt_v[i] != uint32_t(num_v_); ++i) {
      ++s;
    }
  }
  s /= num_v_;
  std::cout << "average normal label size: " << s << std::endl;
}

int PrunedLandmarkLabeling
::dfs(int s, int t, int step, int ele){
  if (step >= constrain) return count;
  // std::cout << "step: " << step << " ele: " << ele << std::endl;
  stack[step] = ele;
  visited[ele] = true;
  for (int i = 0; ; ++i) {
    int w = adj[ele].nb[i];
    // std::cout << "w: " << w << std::endl;
    if (w == num_v_ + 1) break;
    else if (w == t){
      stack[step + 1] = t;
      count++;
    }
    else if (!visited[w]) {
      dfs(s, t, step + 1, w);
    }
  }
  visited[ele] = false;
  return count;
}

int PrunedLandmarkLabeling
::parallel_dfs(int s, int t, int step, int ele){
  omp_set_num_threads(8);
  if (step >= constrain) return count;
  // std::cout << "step: " << step << " ele: " << ele << std::endl;
  for (int i = 0; i < NumThreads; i++){
    count_sum[i] = 0;
  }

  stack[step] = ele;
  visited[ele] = true;

  #pragma omp parallel firstprivate(stack, visited)
  {
    long id = omp_get_thread_num();
    int j;
    for (j = 0; ; ++j) {
      int w = adj[ele].nb[j];
      if (w == num_v_ + 1) break;
    }
  #pragma omp for
    for (int i = 0; i < j ; ++i) {
      int w = adj[ele].nb[i];
      // std::cout << "w: " << w << std::endl;
      if (w == t){
        stack[step + 1] = t;
        count_sum[id]++;
      }
      else if (!visited[w]) {
        para_dfs(s, t, step + 1, w, id);
      }
    }
  }
  visited[ele] = false;

  for (int i = 0; i < NumThreads; i++){
        count += count_sum[i];
  }

  return count;
}

int PrunedLandmarkLabeling
::para_dfs(int s, int t, int step, int ele, long id){
  if (step >= constrain) return count;
  // std::cout << "step: " << step << " ele: " << ele << std::endl;
  stack[step] = ele;
  visited[ele] = true;

  for (int i = 0; ; ++i) {
    int w = adj[ele].nb[i];
    // std::cout << "w: " << w << std::endl;
    if (w == num_v_ + 1) break;
    else if (w == t){
      stack[step + 1] = t;
      count_sum[id]++;
    }
    else if (!visited[w]) {
      para_dfs(s, t, step + 1, w, id);
    }
  }


  visited[ele] = false;
  return count;
}

int PrunedLandmarkLabeling
::DistanceCheck(int s, int t){
  if (s >= num_v_ || t >= num_v_) return s == t ? 0 : INT_MAX;  // INT_MAX = 2147483647
  std::vector<int> que(num_v_);  // queue
  std::vector<bool> vis(num_v_); 
  int que_t0 = 0, que_t1 = 0, que_h = 0;  
  que[que_h++] = s;
  que_t1 = que_h;

  for (uint8_t d = 0; que_t0 < que_h && d <= INF8; ++d) {
    for (int que_i = que_t0; que_i < que_t1; ++que_i) {
      int v = que[que_i];
      if (v == t){
        return d;
      }else{
        for (int i = 0; ; ++i) {
          int w = adj[v].nb[i];
          if (w == num_v_ + 1) break;
          if (!vis[w]) {
            que[que_h++] = w;
            vis[w] = true;
          }
        }
      }
    }
    que_t0 = que_t1;
    que_t1 = que_h;
  }
  return INT_MAX;
}

#endif  // PRUNED_LANDMARK_LABELING_H_
