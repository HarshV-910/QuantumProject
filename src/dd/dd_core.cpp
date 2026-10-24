// Weighted binary decision-diagram quantum state core. C++17, C ABI for ctypes.
#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using C = std::complex<double>;
static constexpr double EPS = 1e-13;
using Edge = struct { C w; int id; };
struct Node { int level; Edge lo, hi; };
static Edge zero_edge(){ return {C(0,0),0}; }
static C clean(C z){ if(std::abs(z.real())<EPS) z.real(0); if(std::abs(z.imag())<EPS) z.imag(0); return z; }
static std::string cq(C z){ z=clean(z); return std::to_string(std::llround(z.real()*1e12))+","+std::to_string(std::llround(z.imag()*1e12)); }
static std::string edgekey(Edge e){ return std::to_string(e.id)+":"+cq(e.w); }

class DD {
public:
 int n; std::vector<Node> nodes; // index 0 is terminal placeholder
 std::unordered_map<std::string,int> unique;
 std::unordered_map<std::string,Edge> addcache, onecache;
 Edge root;
 std::mt19937_64 rng{std::random_device{}()};
 explicit DD(int nn):n(nn),nodes(1),root({C(1,0),0}) { initialize(); }
 void clear(){ addcache.clear(); onecache.clear(); }
 void compact(){
  std::vector<unsigned char> live(nodes.size(),0);
  std::function<void(int)> mark=[&](int id){ if(id<=0 || live[(size_t)id]) return; live[(size_t)id]=1; mark(nodes[(size_t)id].lo.id); mark(nodes[(size_t)id].hi.id); };
  mark(root.id);
  std::vector<int> remap(nodes.size(),0); std::vector<Node> fresh(1); std::unordered_map<std::string,int> fresh_unique;
  for(size_t old=1; old<nodes.size(); ++old){ if(!live[old]) continue; Node x=nodes[old]; x.lo.id=remap[(size_t)x.lo.id]; x.hi.id=remap[(size_t)x.hi.id];
   auto k=nodekey(x.level,x.lo,x.hi); auto it=fresh_unique.find(k); int id;
   if(it==fresh_unique.end()){ id=(int)fresh.size(); fresh.push_back(x); fresh_unique.emplace(k,id); } else id=it->second;
   remap[old]=id;
  }
  root.id=remap[(size_t)root.id]; nodes.swap(fresh); unique.swap(fresh_unique); clear();
 }
 std::string nodekey(int lev, Edge a, Edge b){ return std::to_string(lev)+"|"+edgekey(a)+"|"+edgekey(b); }
 Edge make(int lev, Edge lo, Edge hi){
  lo.w=clean(lo.w); hi.w=clean(hi.w);
  if(std::abs(lo.w)<EPS) lo=zero_edge(); 
  if(std::abs(hi.w)<EPS) hi=zero_edge();
  if(std::abs(lo.w)<EPS && std::abs(hi.w)<EPS) return zero_edge();
  C pivot=std::abs(lo.w)>=EPS?lo.w:hi.w;
  Edge a=std::abs(lo.w)>=EPS?Edge{clean(lo.w/pivot),lo.id}:zero_edge();
  Edge b=std::abs(hi.w)>=EPS?Edge{clean(hi.w/pivot),hi.id}:zero_edge();
  auto k=nodekey(lev,a,b); auto it=unique.find(k); int id;
  if(it==unique.end()){ id=(int)nodes.size(); nodes.push_back({lev,a,b}); unique.emplace(k,id); } else id=it->second;
  return {clean(pivot),id};
 }
 void initialize(){ nodes.assign(1,Node{}); unique.clear(); clear(); Edge e{C(1,0),0}; for(int q=0;q<n;q++) e=make(q,e,zero_edge()); root=e; }
 Edge child(Edge e, bool high){ if(std::abs(e.w)<EPS)return zero_edge(); if(!e.id)return high?zero_edge():e; Node x=nodes[e.id]; Edge c=high?x.hi:x.lo; return {e.w*c.w,c.id}; }
 Edge add(Edge a, Edge b){
  if(std::abs(a.w)<EPS)return b; 
  if(std::abs(b.w)<EPS)return a;
  if(a.id==b.id){C s=clean(a.w+b.w); return std::abs(s)<EPS?zero_edge():Edge{s,a.id};}
  if(!a.id&&!b.id)return {clean(a.w+b.w),0};
  std::string key=edgekey(a)+"+"+edgekey(b); auto it=addcache.find(key); if(it!=addcache.end())return it->second;
  Edge out;
  if(!a.id){Node y=nodes[b.id]; out=make(y.level,add(zero_edge(),child(b,false)),add(zero_edge(),child(b,true)));}
  else if(!b.id){Node x=nodes[a.id]; out=make(x.level,add(child(a,false),zero_edge()),add(child(a,true),zero_edge()));}
  else {Node x=nodes[a.id],y=nodes[b.id]; if(x.level!=y.level)throw std::runtime_error("DD addition level mismatch"); out=make(x.level,add(child(a,false),child(b,false)),add(child(a,true),child(b,true)));}
  addcache.emplace(key,out); return out;
 }
 using M2=std::array<C,4>;
 Edge apply1(Edge e,int q,const M2&m){
  if(std::abs(e.w)<EPS)return zero_edge(); 
  if(!e.id)throw std::runtime_error("single-qubit gate reached terminal");
  Node x=nodes[e.id]; std::string mk; for(C z:m)mk+=cq(z)+";";
  std::string key=std::to_string(e.id)+"/"+cq(e.w)+"/"+std::to_string(q)+"/"+mk;
  auto it=onecache.find(key); if(it!=onecache.end())return it->second;
  Edge lo=child(e,false), hi=child(e,true), out;
  if(x.level>q) out=make(x.level,apply1(lo,q,m),apply1(hi,q,m));
  else if(x.level==q){ Edge a=add(scale(lo,m[0]),scale(hi,m[1])); Edge b=add(scale(lo,m[2]),scale(hi,m[3])); out=make(q,a,b); }
  else throw std::runtime_error("qubit level missing from DD");
  onecache.emplace(key,out); return out;
 }
 Edge scale(Edge e,C s){C w=clean(e.w*s);return std::abs(w)<EPS?zero_edge():Edge{w,e.id};}
 double norm2(Edge e){ if(std::abs(e.w)<EPS)return 0; if(!e.id)return std::norm(e.w); return norm2(child(e,false))+norm2(child(e,true)); }
 double prob(Edge e,int q,int outcome){if(std::abs(e.w)<EPS||!e.id)return 0; Node x=nodes[e.id]; Edge lo=child(e,false),hi=child(e,true); if(x.level==q)return norm2(outcome?hi:lo); if(x.level<q)throw std::runtime_error("measurement qubit missing");return prob(lo,q,outcome)+prob(hi,q,outcome);}
 Edge project(Edge e,int q,int outcome){if(std::abs(e.w)<EPS)return zero_edge(); if(!e.id)throw std::runtime_error("measurement qubit missing"); Node x=nodes[e.id]; Edge lo=child(e,false),hi=child(e,true); if(x.level>q)return make(x.level,project(lo,q,outcome),project(hi,q,outcome)); if(x.level==q)return make(q,outcome?zero_edge():lo,outcome?hi:zero_edge()); throw std::runtime_error("measurement qubit missing");}
 C amp(Edge e,uint64_t idx){if(std::abs(e.w)<EPS)return 0; C w=e.w; int id=e.id; for(int l=n-1;l>=0;l--){if(!id)return 0; Node x=nodes[id]; Edge c=((idx>>l)&1)?x.hi:x.lo; w*=c.w;id=c.id;}return id==0?w:C(0,0);}
 C inner(Edge a,Edge b){if(std::abs(a.w)<EPS||std::abs(b.w)<EPS)return 0; if(!a.id&&!b.id)return std::conj(a.w)*b.w; if(!a.id||!b.id)throw std::runtime_error("inner product level mismatch"); Node x=nodes[a.id],y=nodes[b.id]; if(x.level!=y.level)throw std::runtime_error("inner product level mismatch"); return inner(child(a,false),child(b,false))+inner(child(a,true),child(b,true));}
 void apply2(const std::array<C,16>&m,int ctrl,int tgt){int hi=std::max(ctrl,tgt),lo=std::min(ctrl,tgt); auto bi=[&](int hb,int lb){int cb=(ctrl==hi?hb:lb),tb=(tgt==hi?hb:lb);return 2*cb+tb;};
  std::function<Edge(Edge)> rec=[&](Edge e)->Edge{if(std::abs(e.w)<EPS)return zero_edge(); if(!e.id)throw std::runtime_error("two-qubit gate reached terminal"); Node x=nodes[e.id]; Edge l=child(e,false),h=child(e,true); if(x.level>hi)return make(x.level,rec(l),rec(h)); if(x.level!=hi)throw std::runtime_error("two-qubit gate level missing"); Edge ins[2]={l,h},outs[2]; for(int ob=0;ob<2;ob++){Edge acc=zero_edge();for(int ib=0;ib<2;ib++){M2 block{};for(int ol=0;ol<2;ol++)for(int il=0;il<2;il++)block[2*ol+il]=m[4*bi(ob,ol)+bi(ib,il)];acc=add(acc,apply1(ins[ib],lo,block));}outs[ob]=acc;}return make(hi,outs[0],outs[1]);}; root=rec(root);clear(); }
};
static DD* get(void*p){if(!p)throw std::runtime_error("null DD handle");return static_cast<DD*>(p);}
static DD::M2 gate1(int g,double th,double ph,double lam){double r=1/std::sqrt(2.0); switch(g){case 0:return {r,r,r,-r};case 1:return {0,1,1,0};case 2:return {0,C(0,-1),C(0,1),0};case 3:return {1,0,0,-1};case 4:return {1,0,0,C(0,1)};case 5:return {1,0,0,std::exp(C(0,M_PI/4))};case 6:{double c=cos(th/2),s=sin(th/2);return {c,-std::exp(C(0,lam))*s,std::exp(C(0,ph))*s,std::exp(C(0,ph+lam))*c};}default:throw std::runtime_error("unknown one-qubit gate code");}}
static std::array<C,16> gate2(int g,double th,double ph,double lam){std::array<C,16> m{}; if(g==0){m[0]=m[5]=m[10]=m[15]=1;m[10]=0;m[11]=1;m[14]=1;m[15]=0;} else if(g==1){m[0]=m[5]=m[10]=1;m[15]=-1;} else if(g==2){m[0]=m[15]=1;m[6]=m[9]=1;} else if(g==3){auto u=gate1(6,th,ph,lam);m[0]=m[5]=1;m[10]=u[0];m[11]=u[1];m[14]=u[2];m[15]=u[3];}else throw std::runtime_error("unknown two-qubit gate code");return m;}
extern "C" {
 thread_local std::string dd_last_error;
 void* dd_create(int n){try{if(n<1||n>60)throw std::runtime_error("n must be in [1,60]");return new DD(n);}catch(const std::exception&e){dd_last_error=e.what();return nullptr;}}
 void dd_destroy(void*p){delete static_cast<DD*>(p);}
 const char* dd_error(){return dd_last_error.c_str();}
 int dd_apply_1q(void*p,int g,int q,double th,double ph,double lam){try{DD*d=get(p);if(q<0||q>=d->n)throw std::runtime_error("qubit out of range");d->root=d->apply1(d->root,q,gate1(g,th,ph,lam));d->compact();return 0;}catch(const std::exception&e){dd_last_error=e.what();return -1;}}
 int dd_apply_2q(void*p,int g,int c,int t,double th,double ph,double lam){try{DD*d=get(p);if(c<0||c>=d->n||t<0||t>=d->n||c==t)throw std::runtime_error("invalid control/target qubits");d->apply2(gate2(g,th,ph,lam),c,t);d->compact();return 0;}catch(const std::exception&e){dd_last_error=e.what();return -1;}}
 int dd_measure(void*p,int q){try{DD*d=get(p);if(q<0||q>=d->n)throw std::runtime_error("qubit out of range");double p1=std::clamp(d->prob(d->root,q,1),0.0,1.0);std::uniform_real_distribution<double> dist(0,1);int b=dist(d->rng)<p1;double pr=b?p1:1-p1;if(pr<EPS){b=1-b;pr=b?p1:1-p1;}d->root=d->scale(d->project(d->root,q,b),1/std::sqrt(pr));d->compact();return b;}catch(const std::exception&e){dd_last_error=e.what();return -1;}}
 int dd_sample(void*p,int shots,int*out){try{DD*d=get(p);if(shots<0||!out)throw std::runtime_error("invalid sample arguments");std::uniform_real_distribution<double>dist(0,1);for(int s=0;s<shots;s++){Edge e=d->root;for(int q=d->n-1;q>=0;q--){double p1=std::clamp(d->prob(e,q,1),0.0,1.0);int b=dist(d->rng)<p1;double pr=b?p1:1-p1;if(pr<EPS){b=1-b;pr=b?p1:1-p1;}e=d->scale(d->project(e,q,b),1/std::sqrt(pr));out[s*d->n+q]=b;}}d->compact();return 0;}catch(const std::exception&e){dd_last_error=e.what();return -1;}}
 double dd_expectation(void*p,const char*ps){try{DD*d=get(p);if(!ps||strlen(ps)!=(size_t)d->n)throw std::runtime_error("invalid Pauli string length");Edge e=d->root;for(int q=0;q<d->n;q++){DD::M2 m;switch(ps[q]){case 'I':continue;case 'X':m={0,1,1,0};break;case 'Y':m={0,C(0,-1),C(0,1),0};break;case 'Z':m={1,0,0,-1};break;default:throw std::runtime_error("invalid Pauli character");}e=d->apply1(e,q,m);}double result=d->inner(d->root,e).real();d->compact();return result;}catch(const std::exception&e){dd_last_error=e.what();return NAN;}}
 int dd_node_count(void*p){try{return (int)get(p)->nodes.size()-1;}catch(...){return -1;}}
 int dd_nqubits(void*p){try{return get(p)->n;}catch(...){return -1;}}
 int dd_amplitude(void*p,uint64_t i,double*re,double*im){try{C a=get(p)->amp(get(p)->root,i);*re=a.real();*im=a.imag();return 0;}catch(const std::exception&e){dd_last_error=e.what();return -1;}}
}
