
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
  bool ConstructIndex(std::istream &ifs);
  bool ConstructIndex(const char *filename);

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
      : adj(NULL), time_load_(0), time_indexing_(0), num_v_(0), visited(NULL), count(0){}
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
  int stack[constrain + 1]; 
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
  std::vector<std::vector<int> > adj_out(V);
  for (size_t i = 0; i < es.size(); ++i) {
    int v = es[i].first, w = es[i].second;
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

void PrunedLandmarkLabeling
::Free() {
  if (num_v_ != 0){
    for (int v = 0; v < num_v_; ++v) {
      free(adj[v].nb);
    }
    free(adj);
    free(visited);
    adj = NULL;
    visited = NULL;
    num_v_ = 0;
  }
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
  omp_set_num_threads(NumThreads);
  if (step >= constrain) return count;
  // std::cout << "step: " << step << " ele: " << ele << std::endl;
  for (int i = 0; i < NumThreads; i++){
    count_sum[i] = 0;
  }

  free(visited);
  
  // stack[step] = ele; firstprivate(stack) 

  #pragma omp parallel private(visited)
  {
    long id = omp_get_thread_num();
    int j;
    for (j = 0; ; ++j) {
      int w = adj[ele].nb[j];
      if (w == num_v_ + 1) break;
    }
  #pragma omp for
    for (int i = 0; i < j ; ++i) {

      visited = (bool*)memalign(64, num_v_ * sizeof(bool));
      for (int v = 0; v < num_v_; v++){
        visited[v] = false;
      }
      visited[ele] = true;

      int w = adj[ele].nb[i];
      // std::cout << "w: " << w << std::endl;
      if (w == t){
        // stack[step + 1] = t;
        count_sum[id]++;
      }
      else if (!visited[w]) {
        para_dfs(s, t, step + 1, w, id);
      }
      visited[ele] = false;
      free(visited);
    }
  }
  for (int i = 0; i < NumThreads; i++){
        count += count_sum[i];
  }

  return count;
}

int PrunedLandmarkLabeling
::para_dfs(int s, int t, int step, int ele, long id){
  if (step >= constrain) return count;
  // std::cout << "step: " << step << " ele: " << ele << std::endl;
  // stack[step] = ele;
  visited[ele] = true;

  for (int i = 0; ; ++i) {
    int w = adj[ele].nb[i];
    // std::cout << "w: " << w << std::endl;
    if (w == num_v_ + 1) break;
    else if (w == t){
      // stack[step + 1] = t;
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
