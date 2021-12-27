#include <sys/time.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "pruned_landmark_labeling.h"
#include "io.h"

using namespace std;

// const int kNumQueries = 1000;

double GetCurrentTimeSec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "usage: construct_index GRAPH" << endl;
  }

  PrunedLandmarkLabeling pll;
  if (!pll.ConstructIndex(argv[1])) {
    cerr << "error: Load failed" << endl;
    exit(EXIT_FAILURE);
  }
  pll.PrintStatistics();

  int QueryNum = 1000;
  vector<int> vs(QueryNum), ws(QueryNum);
  for (int i = 0; i < QueryNum; ++i) {
    vs[i] = rand() % pll.GetNumVertices();
    ws[i] = rand() % pll.GetNumVertices();
  }

  // std::vector<std::pair<uint32_t, uint32_t>> queries;
  // IO::read("samples/unhot2hot_pairs.bin", queries);
  // cout << "finish reading file" << endl;

  // uint32_t QueryNum = queries.size();
  for (int i = 0; i < QueryNum; ++i) {
    cout << vs[i] << "->" << ws[i] << endl;
    double time_start = GetCurrentTimeSec();
    // pll.BuildBigraph(vs[i], ws[i]);
    double elapsed_time = GetCurrentTimeSec() - time_start;
    cout << i << ": index duration: " << elapsed_time << endl;

    time_start = GetCurrentTimeSec();
    // auto query = queries[i];
    pll.count = 0;
    // cout << query.first << "->" << query.second << endl;
    // pll.dfs(query.first, query.second, 0, query.first);
    pll.dfs(vs[i],ws[i],0,vs[i]);
    elapsed_time = GetCurrentTimeSec() - time_start;
    cout << i << ": " << pll.count << " dfs duration: " << elapsed_time << endl;
    
    
    for (int i = 0; i < NumThreads; i++){
      pll.count_sum[i] = 0;
    }
    pll.count = 0;
    time_start = GetCurrentTimeSec();
    // pll.parallel_dfs(query.first, query.second, 0, query.first);
    pll.parallel_dfs(vs[i],ws[i],0,vs[i]);
    elapsed_time = GetCurrentTimeSec() - time_start;
    cout << i << ": " << pll.count << " parallel dfs duration: " << elapsed_time << endl;
  }
  // double elapsed_time = GetCurrentTimeSec() - time_start;
  // cout << "average query time: "
  //      << elapsed_time / QueryNum * 1E6
  //      << " microseconds" << endl;



  
  // time_start = GetCurrentTimeSec();
  // for (uint32_t i = 2; i < QueryNum; ++i) {
  //   auto query = queries[i];
  //   pll.count = 0;
  //   pll.parallel_dfs(query.first, query.second, 0, query.first);
  //   cout << i << ": " << pll.count << endl;
  // }
  // elapsed_time = GetCurrentTimeSec() - time_start;
  // cout << "average query time: "
  //      << elapsed_time / QueryNum * 1E6
  //      << " microseconds" << endl;
  // double correctness = ((double) kNumQueries - (double) count) / (double) kNumQueries;
  // cout << "correctness: " << correctness<< endl;
  exit(EXIT_SUCCESS);
}
