# 数据结构

### 树状数组

```cpp
int tr[N];

int lowbit(int x)
{
    return x&-x;
}

void add(int x,int c) // 第x位加上c
{
    for(int i = x;i<=n;i+=lowbit(i))
    {
        tr[i]+=c;
    }
}

int sum(int x) // 求前x位的和
{
    int res = 0;
    for(int i = x;i;i-=lowbit(i))
    {
        res+=tr[i];
    }
    return res;
}
```

### 线段树（处理区间和）

```cpp
struct node
{
    int l, r;
    ll sum, add;
} tr[N * 4];

void pushup(int u)
{
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}

void pushdown(int u)
{
    auto &root = tr[u], &left = tr[u << 1], &right = tr[u << 1 | 1];
    if (root.add)
    {
        left.add += root.add;
        left.sum += (left.r - left.l + 1) * root.add;
        right.add += root.add;
        right.sum += (right.r - right.l + 1) * root.add;
        root.add = 0;
    }
}

void bulid(int u, int l, int r)
{
    if (l == r)
    {
        tr[u] = {l, r, a[l], 0};
    }
    else
    {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        bulid(u << 1, l, mid);
        bulid(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

void modify(int u, int l, int r, ll v)
{
    if (tr[u].l >= l && tr[u].r <= r)
    {
        tr[u].sum += (ll)(tr[u].r - tr[u].l + 1) * v;
        tr[u].add += v;
    }
    else
    {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid)
            modify(u << 1, l, r, v);
        if (r > mid)
            modify(u << 1 | 1, l, r, v);
        pushup(u);
    }
}

ll query(int u, int l, int r)
{
    if (tr[u].l >= l && tr[u].r <= r)
        return tr[u].sum;
    else
    {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        ll sum = 0;
        if (l <= mid)
            sum += query(u << 1, l, r);
        if (r > mid)
            sum += query(u << 1 | 1, l, r);
        return sum;
    }
}
```

### 莫队算法

对于区间 $[l,r]$ ，以 $l$ 所在块的编号为第一关键字排序，以 $r$ 为第二关键字排序。

```
bool cmp1(const query a, const query b)
{
    if (a.pos == b.pos)
        return a.r < b.r;
    return a.pos < b.pos;
}

int len = sqrt(n);
for (int i = 1; i <= m; i++)
{
    cin >> q[i].l >> q[i].r;
    q[i].id = i;
    q[i].pos = q[i].l / len;
}
sort(q + 1, q + 1 + m, cmp1);
for (int i = 1, l = 1, r = 0; i <= m; i++)
{
    while (l < q[i].l)sub(l++); 
    while (l > q[i].l)add(--l);
    while (r < q[i].r)add(++r);
    while (r > q[i].r)sub(r--); 
    q[i].ans = res;
}
```

### 树上启发式合并

```
void dfs(int u,int f)           //与重链剖分相同的写法找重儿子
{
    siz[u]=1;
    for(int i=Head[u];~i;i=Edge[i].next)
    {
        int v = Edge[i].to;
        if(v==f) continue;
        dfs(v,u);
        siz[u]+=siz[v];
        if(siz[v]>siz[son[u]])
            son[u]=v;
    }
}
int col[maxn],cnt[maxn];    //col存放某结点的颜色，cnt存放某颜色在“当前”子树中的数量
long long ans[maxn],sum;    //ans是答案数组，sum用于累加计算出“当前”子树的答案
int flag,maxc;              //flag用于标记重儿子，maxc用于更新最大值
//TODO 统计某结点及其所有轻儿子的贡献
void count(int u,int f,int val)
{
    cnt[col[u]]+=val;//val为正为负可以控制是增加贡献还是删除贡献
    if(cnt[col[u]]>maxc)    //找最大值，基操吧
    {
        maxc=cnt[col[u]];
        sum=col[u];
    }
    else if(cnt[col[u]]==maxc)  //这样做是因为如果两个颜色数量相同那都得算
        sum+=col[u];
    for(int i=Head[u];~i;i=Edge[i].next) //排除被标记的重儿子，统计其它儿子子树信息
    {
        int v = Edge[i].to;
        if(v==f||v==flag) continue; //不能写if(v==f||v==son[u]) continue;
        count(v,u,val);
    }
}
//dsu on tree的板子
void dfs(int u,int f,bool keep)
{
    // 第一步：搞轻儿子及其子树算其答案删贡献
    for(int i=Head[u];~i;i=Edge[i].next)
    {
        int v = Edge[i].to;
        if(v==f||v==son[u]) continue;
        dfs(v,u,false);
    }
    // 第二步：搞重儿子及其子树算其答案不删贡献
    if(son[u])
    {
        dfs(son[u],u,true);
        flag = son[u];
    }
    // 第三步：暴力统计u及其所有轻儿子的贡献合并到刚算出的重儿子信息里
    count(u,f,1);
    flag = 0;   
    ans[u] = sum;
    // 把需要删除贡献的删一删
    if(!keep)
    {
        count(u,f,-1);
        sum=maxc=0; //这是因为count函数中会改变这两个变量值
    }
}
```



### 主席树

```cpp
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define ll long long
#define PII pair<int, int>
#define pi acos(-1.0)
const int N = 1e5 + 10;
int n, q;
int a[N];
vector<int> t;
int ls[N * 25], rs[N * 25], idx, cnt[N * 25], root[N * 25];

int find(int x)
{
    return lower_bound(t.begin(), t.end(), x) - t.begin() + 1;
}

void modify(int &cur, int past, int x, int l, int r)
{
    cur = ++idx;
    ls[cur] = ls[past];
    rs[cur] = rs[past];
    cnt[cur] = cnt[past] + 1;
    if (l == r)
        return;
    int mid = l + r >> 1;
    if (x <= mid)
        modify(ls[cur], ls[past], x, l, mid);
    else
        modify(rs[cur], rs[past], x, mid + 1, r);
}

int query(int lx, int rx, int l, int r, double v)
{
    if (l == r)
        return t[l - 1];
    int mid = l + r >> 1;
    int res = -1;
    if ((double)(cnt[ls[rx]]) - (double)(cnt[ls[lx]]) > v)
        res = query(ls[lx], ls[rx], l, mid, v);
    if (res == -1 && (double)(cnt[rs[rx]]) - (double)(cnt[rs[lx]]) > v)
        res = query(rs[lx], rs[rx], mid + 1, r, v);
    return res;
}

int main()
{
    // ios::sync_with_stdio(false);cin.tie(0);
    cin >> n >> q;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        t.push_back(a[i]);
    }
    sort(t.begin(), t.end());
    t.erase(unique(t.begin(), t.end()), t.end());
    for (int i = 1; i <= n; i++)
    {
        modify(root[i], root[i - 1], find(a[i]), 1, t.size());
    }
    while (q--)
    {
        int l, r, k;
        cin >> l >> r >> k;
        cout << query(root[l - 1], root[r], 1, t.size(), (double)(r - l + 1) / (double)(k)) << endl;
    }
    return 0;
}
```

