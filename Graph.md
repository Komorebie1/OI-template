## 图论

### 链式前向星

```cpp
// 对于每个点k，开一个单链表，存储k所有可以走到的点。h[k]存储这个单链表的头结点
int h[N], e[N], ne[N], idx;

// 添加一条边a->b
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}
```

### 拓扑排序

```cpp
bool topsort()
{
    int hh = 0, tt = -1;
    // d[i] 存储点i的入度
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;
    while (hh <= tt)
    {
        int u = q[hh ++ ];

        for (int v:G[u])
        {
            if (-- d[v] == 0)
                q[ ++ tt] = v;
        }
    }
    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
    return tt == n - 1;
}
```

### dijkstra

###### 朴素版dijkstra

时间复杂度 $O(n^2+m)$ .

```cpp
int g[N][N];  // 存储每条边
int dist[N];  // 存储1号点到每个点的最短距离
bool st[N];   // 存储每个点的最短路是否已经确定

// 求1号点到n号点的最短路，如果不存在则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < n - 1; i ++ )
    {
        int t = -1;     // 在还未确定最短路的点中，寻找距离最小的点
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        // 用t更新其他点的距离
        for (int j = 1; j <= n; j ++ )
            dist[j] = min(dist[j], dist[t] + g[t][j]);

        st[t] = true;
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```

###### 堆优化版dijkstra

时间复杂度 $O(m \log n)$. 

```cpp
int dist[N];        // 存储所有点到1号点的距离
bool st[N];     // 存储每个点的最短距离是否已确定
// 求1号点到n号点的最短距离，如果不存在，则返回-1
int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});  // first存储距离，second存储节点编号
    while (heap.size())
    {
        auto [d, u] = heap.top();
        heap.pop();
        if (st[u]) continue;
        st[u] = true;

        for (auto [v, w] : G[u])
        {
            if (dist[v] > distance + w)
            {
                dist[v] = distance + w;
                heap.push({dist[v], v});
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```

### Bellman-Ford

时间复杂度 $O(mn)$ .

```cpp
struct EDGE // for bellman-ford
{
    int u, int v, int w;
};

vector<EDGE> edges;

bool Bellman_ford(int n, int s)
{
    memset(dis, 0x3f, sizeof(dis));
    dis[s] = 0;
    bool flag;
    for (int i = 1; i <= n; i++)
    {
        flag = false;
        for (int j = 0; j < edges.size(); j++)
        {
            int u = edges[j].u, v = edges[j].v, w = edges[j].w;
            if (dis[u] == inf)
                continue;
            if (dis[v] > dis[u] + w)
            {
                dis[v] = dis[u] + w;
                flag = true;
            }
        }
        if (!flag)
            break;
    }
    return flag; // 第n轮循环仍可以松弛说明存在负环
}
```

### SPFA（队列优化版Bellman-Ford、自带负环判断）

平均时间复杂度 $O(m)$ ，最坏 $O(nm)$ .

```cpp
int vis[N], cnt[N];
vector<PII> G[N];

bool spfa(int n, int s)
{
    memset(dis, 0x3f, sizeof(dis));
    dis[s] = 0;
    queue<int> q;
    while (q.size())
    {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (auto [v, w] : G[u])
        {
            if (dis[v] >= dis[u] + w)
            {
                dis[v] = dis[u] + w;
                cnt[v] = cnt[u] + 1;  // 记录经过了多少点
                if (cnt[v] >= n)      // 存在负环
                    return false;
                if (!vis[v])
                {
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return true;
}
```

### Kruskal

时间复杂度 $O(mlogm)$ .

```cpp
int n, m;       // n是点数，m是边数
int p[N];       // 并查集的父节点数组

struct Edge     // 存储边
{
    int a, b, w;

    bool operator< (const Edge &W)const
    {
        return w < W.w;
    }
}edges[M];

int find(int x)     // 并查集核心操作
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

int kruskal()
{
    sort(edges, edges + m);

    for (int i = 1; i <= n; i ++ ) p[i] = i;    // 初始化并查集

    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;

        a = find(a), b = find(b);
        if (a != b)     // 如果两个连通块不连通，则将这两个连通块合并
        {
            p[a] = b;
            res += w;
            cnt ++ ;
        }
    }

    if (cnt < n - 1) return INF;
    return res;
}
```

### 染色法判断二分图

时间复杂度 $O(n+m)$ .

```cpp
int n;                       // n表示点数
int color[N];                // 表示每个点的颜色，-1表示为染色，0表示白色，1表示黑色

// 参数：u表示当前节点，c表示当前点的颜色
bool dfs(int u, int c)
{
    color[u] = c;
    for (int v : G[u])
    {
        if (color[v] == -1)
        {
            if (!dfs(v, !c)) return false;
        }
        else if (color[v] == c)
            return false;
    }

    return true;
}

bool check()
{
    memset(color, -1, sizeof color);
    bool flag = true;
    for (int i = 1; i <= n; i++)
        if (color[i] == -1)
            if (!dfs(i, 0))
            {
                flag = false;
                break;
            }
    return flag;
}
```

### 匈牙利算法求最大匹配

时间复杂度$O(nm)$.

```cpp
int n1, n2;        // n1表示第一个集合中的点数，n2表示第二个集合中的点数
vector<int> G[N];  // 匈牙利算法中只会用到从第二个集合指向第一个集合的边，所以这里只用存一个方向的边
int match[N];      // 存储第二个集合中的每个点当前匹配的第一个集合中的点是哪个
bool st[N];        // 表示第二个集合中的每个点是否已经被遍历过

bool find(int u)
{
    for (int v : G[u])
    {
        if (!st[v])
        {
            st[v] = true;
            if (match[v] == 0 || find(match[v]))
            {
                match[v] = u;
                return true;
            }
        }
    }
    return false;
}

int res = 0;
for (int i = 1; i <= n1; i++)  // 求最大匹配数，依次枚举第一个集合中的每个点能否匹配第二个集合中的点
{
    memset(st, false, sizeof st);
    if (find(i)) res++;
}
```

### 树的直径

```cpp
//两遍dfs
int first[N], next[N], aim[N], idx;
int diameter[N], c;
void dfs(int u, int fa)
{
    for (int i = first[u]; i; i = next[i])
    {
        if (aim[i] == fa)
            continue;
        diameter[aim[i]] = diameter[u] + 1;
        if (diameter[aim[i]] > diameter[c])
            c = aim[i];
        dfs(aim[i], u);
    }
}
```

```cpp
//求树的直径（树形dp法）//边权全为正，所有直径的中点重合
int d1[N], d2[N], diameter;
void dfs(int u, int fa) //两个数组实现
{
    d1[u] = d1[u] = 0;
    int d = 0;
    for (int i = first[u]; i; i = next[i])
    {
        if (aim[i] == fa)
            continue;
        dfs(aim[i], u);
        d = d1[aim[i]] + 1;
        if (d > d1[u])
        {
            d2[u] = d1[u];
            d1[u] = d;
        }
        else if (d > d2[u])
        {
            d2[u] = d;
        }
    }
    diameter = max(diameter, d1[u] + d2[u]);
}
```

```cpp
void dfs(int u, int fa)  // 树形dp法 一个数组实现 存的是到子树中的最长路径
{
    for (int v : G[u])
    {
        if (v == fa) continue;
        dfs(v, u);
        diameter = max(diameter, d1[u] + d1[v] + 1);
        d1[u] = max(d1[u], d1[v] + 1);
    }
}
```

### 树的重心

```cpp
vector<int> centroid;
int sz[N], weight[N];// weight记录子树大小最大值

void dfs(int u, int fa)
{
    sz[u] = 1, weight[u] = 0;
    for (auto v : G[u])
    {
        if (v == fa) continue;
        dfs(v, u);
        sz[u] += sz[v];
        weight[u] = max(weight[u], sz[v]);
    }
    weight[u] = max(n - weight[u], weight[u]);
    if (weight[u] <= n / 2)  //所有子树大小都不超过n/2
    {
        centroid.push_back(u);
    }
}
```

### 最近公共祖先

```cpp
//树上LCA
vector<int> e[N];
vector<int> w[N];
int fa[N][31], cost[N][31], dep[N], dist[N];
void dfs(int u, int f)
{
    fa[u][0] = f;
    dep[u] = dep[f] + 1;
    for (int i = 1; i < 31; i++)
    {
        fa[u][i] = fa[fa[u][i - 1]][i - 1];
        // cost[u][i] = cost[fa[u][i-1]][i-1]+ cost[u][i-1];
    }
    for (int v:e[u])
    {
        if (v != f)
        {
            // cost[e[i]][0] = w[u][i];
            dist[v] = dist[u] + w[u][i];
            dfs(v, u);
        }
    }
}

int lca(int x, int y)
{
    if (dep[x] < dep[y])
    {
        swap(x, y);
    }
    int d = dep[x] - dep[y];
    for (int i = 0; (1 << i) <= d; i++)
    {
        if ((d >> i) & 1)
            x = fa[x][i];
    }
    if (x == y)
        return x;
    for (int i = log2(dep[y]); i >= 0; i--)
    {
        if (fa[x][i] != fa[y][i])
        {
            x = fa[x][i];
            y = fa[y][i];
        }
    }
    return fa[x][0];
}

void solve(int x, int y)
{
    dfs(1, 0);
    int LCA = lca(x, y);
    // cout<<dist[x]+dist[y]-2*dist[LCA]>>endl;//两点最短距离
}
```

### tarjan求割点

$low$：最多经过一条**后向边**能追溯到的最小树中结点编号。

一个顶点 $u$ 是割点，当且仅当满足(1)或(2)：

1. $u$ 为树根，且 $u$ 有多于一个子树。因为无向图 $DFS$ 搜索树中不存在横叉边，所以若有多个子树，这些子树间不会有边相连。

2. $u $不为树根，且满足存在 $(u,v)$ 为树枝边（即 $u$ 为 $v$ 在搜索树中的父亲），并使得 $DFN(u)<=Low(v)$ .（因为删去 $u$ 后 $v$ 以及 $v$ 的子树不能到达 $u$ 的其他子树以及祖先）

```c++
int dfn[N], low[N], tim, vis[N], flag[N];
int ans;
vector<int> G[N];

void dfs(int u, int fa)
{
    dfn[u] = low[u] = ++tim;
    vis[u] = 1;
    int children = 0;
    for (int v : G[u])
    {
        if (!vis[v])
        {
            children++;
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            if (u != fa && low[v] >= dfn[u] && !flag[u])
            {
                flag[u] = 1;
                ans++;
            }
        }
        else if (v != fa)
        {
            low[u] = min(low[u], dfn[v]);
        }
    }
    if (u == fa && children >= 2 && !flag[u])
    {
        flag[u] = 1;
        ans++;
    }
}
```

### tarjan求强连通分量

$low$：最多经过一条**后向边或栈中横插边**所能到达的栈中的最小编号。

```c++
stack<int>st;
int in_stack[N];

void tarjan(int u)
{
    low[u] = dfn[u] = ++idx;
    st.push(u);
    in_stack[u] = 1;
    for(int v:G[u])
    {
        if(!dfn[v])
        {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }
        else if(in_stack[v])
        {
            low[u] = min(low[u], dfn[v]);
        }
    }
    if(low[u] == dfn[u])
    {
        ++sc;
        while(st.top()!=u)
        {
            scc[st.top()] = sc;
            in_stack[st.top()] = 0;
            st.pop();
            sz[sc]++;
        }
        scc[st.top()] = sc;
        in_stack[st.top()] = 0;
        st.pop();
        sz[sc]++;
    }
}
```

### tarjan求点双连通分量

```c++
vector<int>G[N]; // 原图
vector<int>T[N]; // 新图（圆方树）
void tarjan(int u, int fa)
{
    int son = 0;
    dfn[u] = low[u] = ++tim;
    st.push(u);
    for (int v : G[u])
    {
        if (!dfn[v])
        {
            son ++;
            tarjan(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] >= dfn[u])
            {
                scc++;
                while (st.top() != v)
                {
                    ans[scc].push_back(st.top());
                    T[scc].push_back(st.top());
                    T[st.top()].push_back(scc);
                    st.pop();
                }
                ans[scc].push_back(st.top());
                T[scc].push_back(st.top());
                T[st.top()].push_back(scc);
                st.pop();
                ans[scc].push_back(u);
                T[scc].push_back(u);
                T[u].push_back(scc);
            }
        }
        else if (v != fa)//返祖边
        {
            low[u] = min(low[u], dfn[v]);
		}
    }
    if (fa == 0 && son == 0)
        ans[++scc].push_back(u);//特判孤立点
}
```

### tarjan求边双连通分量

```cpp
void tarjan(int u, int fa)
{
    dfn[u] = low[u] = ++tim;
    int son = 0;
    for (int v : G[u])
    {
        if (!dfn[v])
        {
            son++;
            tarjan2(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] > dfn[u]) // 找割边
            {
                cnt_bridge++;
                es[mp[hh(u, v)]].tag = 1;
            }
        }
        else if (v != fa)
        {
            low[u] = min(low[u], dfn[v]);
        }
    }
}
```

### EK求最大流

建边

```cpp
memset(h, -1, sizeof(h));//将h初始化为-1，因为边的标号从0开始。

void add(int a, int b, int c)
{
    e[idx] = b, ne[idx] = h[a], f[idx] = c, h[a] = idx++;
    e[idx] = a, ne[idx] = h[b], f[idx] = 0, h[b] = idx++;
}
```

时间复杂度：$O(nm^2)$

```cpp
bool bfs()
{
    queue<int> q;
    memset(st, false, sizeof(st));
    q.push(S);
    st[S] = true;
    d[S] = inf;
    while (q.size())
    {
        int u = q.front();
        q.pop();
        for (int i = h[u]; i != -1; i = ne[i])
        {
            int v = e[i];
            if (!st[v] && f[i])
            {
                st[v] = true;
                d[v] = min(d[u], f[i]);
                pre[v] = i;
                if (v == T)
                    return true;
                q.push(v);
            }
        }
    }
    return false;
}

int EK()
{
    int r = 0;
    while (bfs())
    {
        r += d[T];
        for (int i = T; i != S; i = e[pre[i ^ 1]])
        {
            f[pre[i]] -= d[T];
            f[pre[i ^ 1]] += d[T];
        }
    }
    return r;
}
```

### Dinic求最大流

时间复杂度： $O(n^2m)$

```cpp
int d[N]; //记录层数

bool bfs()
{
    memset(d, 0, sizeof(d));
    queue<int> q;
    q.push(S);
    dis[S] = 0;
    cur[S] = h[S];
    while (q.size())
    {
        int u = q.front();
        q.pop();
        for (int i = h[u]; ~i; i = ne[i])
        {
            int v = e[i];
            if (d[v] == -1 && f[i])
            {
                d[v] = d[u] + 1;
                cur[v] = h[v];
                if (v == T)
                    return true;
                q.push(v);
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T)
        return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int v = e[i];
        if (d[v] == d[u] + 1 && f[i])
        {
            int t = find(v, min(limit - flow, f[i]));
            if (!t)
                d[v] = -1; //important!!!!!!!!!!!!!!!!!!!!!!!!!!
            f[i] -= t, f[i ^ 1] += t;
            flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs())
    {
        while (flow = find(S, INF))
        {
            r += flow;
        }
    }
    return r;
}
```

### 最小费用最大流（EK算法）

建边

```
void add(int a, int b, int c, int d)
{
    ne[idx] = h[a], e[idx] = b, f[idx] = c, w[idx] = d, h[a] = idx++;
    ne[idx] = h[b], e[idx] = a, f[idx] = 0, w[idx] = -d, h[b] = idx++;
}
```

时间复杂度：$O(nm^2)$

```cpp
bool spfa()
{
    queue<int> q;
    memset(dis, 0x3f3f3f3f, sizeof(dis));
    memset(incf, 0, sizeof(incf));
    q.push(S);
    dis[S] = 0;
    incf[S] = 0x3f3f3f3f;
    while (q.size())
    {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (int i = h[u]; ~i; i = ne[i])
        {
            int v = e[i];
            if (f[i] && dis[v] > dis[u] + w[i])
            {
                dis[v] = dis[u] + w[i];
                pre[v] = i;
                incf[v] = min(f[i], incf[u]);
                if (vis[v] == 0)
                {
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return incf[T] > 0;
}

void EK(int &flow, int &cost)
{
    flow = cost = 0;
    while (spfa())
    {
        int t = incf[T];
        flow += t;
        cost += t * dis[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1])
        {
            f[pre[i]] -= t;
            f[pre[i] ^ 1] += t;
        }
    }
}

```

### 2-sat

选了 $a$ 必须选 $b$，则有 $a\to {b}$​。

其实就是建好图之后求出每个点的正状态和反状态的强连通分量，如果在一个强连通分量则 $false$。