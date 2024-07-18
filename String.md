## 字符串

### KMP

```cpp
// 求Next数组：
// s[]是模式串，p[]是模板串, n是s的长度，m是p的长度
for (int i = 2, j = 0; i <= m; i ++ )
{
    while (j && p[i] != p[j + 1]) j = ne[j];
    if (p[i] == p[j + 1]) j ++ ;
    ne[i] = j;
}
```

```cpp
// 匹配
for (int i = 1, j = 0; i <= n; i ++ )
{
    while (j && s[i] != p[j + 1]) j = ne[j];
    if (s[i] == p[j + 1]) j ++ ;
    if (j == m)
    {
        j = ne[j];
        // 匹配成功后的逻辑
    }
}
```

### Manacher

```cpp
//Manacher（马拉车）求回文半径
int d1[N],d2[N];//d1表示奇数半径（包括自己）
for (int i = 1, l = 1, r = 0; i <= n; i++)
{
    int k = i > r ? 1 : min(d1[l + (r - i)], r - (i - 1));
    while (1 <= i - k && i + k <= n && str[i - k] == str[i + k])
    {
        k++;
    }
    d1[i] = k--;
    if (i + k > r)
    {
        l = i - k;
        r = i + k;
    }
}

for (int i = 1, l = 1, r = 0; i <= n; i++)
{
    int k = i > r ? 1 : min(d2[l + r - i + 1], r - (i - 1));
    while (1 <= i - k - 1 && i + k <= n && str[i - k - 1] == str[i + k])
    {
        k++;
    }
    d2[i] = k--;
    if (i + k > r)
    {
        l = i - k - 1;
        r = i + k;
    }
}
```

### Trie树

```cpp
int son[N][26], cnt[N], idx;
// 0号点既是根节点，又是空节点
// son[][]存储树中每个节点的子节点
// cnt[]存储以每个节点结尾的单词数量

// 插入一个字符串
void insert(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;
}

// 查询字符串出现的次数
int query(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}
```

### AC自动机

```cpp
#include <bits/stdc++.h>
using namespace std;

int main()
{
    ios::sync_with_stdio(false);cin.tie(nullptr);
    int n;
    cin >> n;
    vector<string> T(n);
    string S;
    for (auto &T : T)
        cin >> T;
    cin >> S;
    int trie_size = 10;
    for (auto T : T)
        trie_size += T.size();
    vector<vector<int>> trie(trie_size, vector<int>(26, 0));
    vector<int> pos(n, 0);
    vector<int> fail(trie_size, 0);
    
    // trie树
    int idx = 1;
    for (int i = 0; i < n; i++){
        int p = 1;
        for (auto c : T[i]){
            int cur = c - 'a';
            if (!trie[p][cur])
                trie[p][cur] = ++idx;
            p = trie[p][cur];
        }
        pos[i] = p;
    }
	// 处理根节点的回跳
    vector<int> q;
    int ql = 0;
    for (auto &c : trie[1]){
        if (c){
            fail[c] = 1;
            q.push_back(c);
        }
        else
            c = 1;
    }
	// BFS
    while (ql < q.size())
    {
        int u = q[ql++];
        for (int c = 0; c < 26; c++)
        {
            if (trie[u][c]) // 有儿子存在时
            {
                fail[trie[u][c]] = trie[fail[u]][c]; // 回跳边（四边形）
                q.push_back(trie[u][c]);
            }
            else
            {
                trie[u][c] = trie[fail[u]][c]; // 转移边（三角形）
            }
        }
    }

    vector<int> cnt(trie_size, 0);
	// 跑一遍主串
    for (int cur = 1, i = 0; i < S.size(); i++){
        int nxt = trie[cur][S[i] - 'a'];
        cnt[cur = nxt]++;
    }
	// 按照BFS序倒着遍历，往上传递标记
    reverse(q.begin(), q.end());
    for (auto cur : q){
        cnt[fail[cur]] += cnt[cur];
    }

    for (int i = 0; i < n; i++){
        cout << cnt[pos[i]] << endl;
    }

    return 0;
}
```

### Z函数（扩展KMP）

$z[i] = lcp(s[1:n-1], s[i: n-1])$

```cpp
vector<int> get_z(string s)
{
    int n = s.size();
    vector<int> z(n);
    for (int i = 1, l = 0; i < n; i++)
    {
        if (i <= l + z[l] - 1) z[i] = min(z[i - l], l + z[l] - i);
        while (i + z[i] < n && s[i + z[i]] == s[z[i]]) z[i]++;
        if (i + z[i] > l + z[l]) l = i;
    }
    return z;
}//最后需要修改z[0] = n;
```

### 后缀排序

$sa$ 表示排第 $i$ 的后缀的前一半是第几个后缀, $sa2$ 表示排第$i$的后缀的后一半是第几个后缀，$rk$ 表示第 $i$ 个后缀排第几, 有$sa[rk[i]]=i, rk[sa[i]]=i$. 

$height[i]: lcp(sa[i],sa[i−1]) $，即排名为 $i$ 的后缀与排名为 $i−1$ 的后缀的最长公共前缀。

$height[rak[i]]$，即 $i$​ 号后缀与它前一名的后缀的最长公共前缀。

经典应用：

- 两个后缀的最大公共前缀：$lcp(x,y)=min(heigh[x-y])$，用RMQ维护，$O(1)$ 查询。
- 可重叠最长重复子串：$height$ 数组中最大值
- 本质不同的字串的数量：枚举每一个后缀，第 $i$ 个后缀对答案的贡献为 $len-sa[i]+1-height[i]$

```cpp
int sa[N], rk_base[2][N], *rk = rk_base[0], *rk2 = rk_base[1], sa2[N], cnt[N], height[N];

void get_sa(const char* s, int n)
{
    int m = 122;
    for (int i = 0; i <= m; ++i) cnt[i] = 0;
    for (int i = 1; i <= n; ++i) cnt[rk[i] = s[i]] += 1;
    for (int i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
    for (int i = 1; i <= n; ++i) sa[cnt[rk[i]]--] = i;
    for (int d = 1; d <= n; d <<= 1)
    {
        // 按第二关键字排序
        int p = 0;
        for (int i = n - d + 1; i <= n; i++) sa2[++p] = i;
        for (int i = 1; i <= n; ++i)
            if (sa[i] > d) sa2[++p] = sa[i] - d;
        // 按第一关键字排序
        for (int i = 0; i <= m; ++i) cnt[i] = 0;
        for (int i = 1; i <= n; ++i) cnt[rk[i]] += 1;
        for (int i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
        for (int i = n; i; --i) sa[cnt[rk[sa2[i]]]--] = sa2[i];

        rk2[sa[1]] = 1;
        for (int i = 2; i <= n; ++i) rk2[sa[i]] = rk2[sa[i - 1]] + (rk[sa[i]] != rk[sa[i - 1]] || rk[sa[i] + d] != rk[sa[i - 1] + d]);

        std::swap(rk, rk2);

        m = rk[sa[n]];
        if (m == n) break;
    }
}

void get_height()
{
    for (int i = 1; i <= n; i++) rk[sa[i]] = i;
    for (int i = 1, k = 0; i <= n; i++)
    {
        if (rk[i] == 1) continue;
        if (k) k--;
        int j = sa[rk[i] - 1];
        while (s[i + k] == s[j + k]) k++;
        height[rk[i]] = k;
    }
}
```

### 回文自动机

```cpp
struct PAM {
    int len[N], num[N], fail[N], trie[N][26], tot = 1;
    int getfail(int x, int i, string s)
    {
        while (i - len[x] - 1 < 0 || s[i] != s[i - len[x] - 1]) x = fail[x];
        return x;
    }
    void build(string s)
    {
        int cur = 0;
        fail[0] = 1, len[1] = -1;
        for (int i = 0; i < s.size(); i++)
        {
            int u = s[i] - 'A';
            int pos = getfail(cur, i, s);
            if (!trie[pos][u])
            {
                trie[pos][s[i]] = ++tot;
                fail[tot] = trie[getfail(fail[pos], i, s)][u];
                len[tot] = len[pos] + 2;
            }
            cur = trie[pos][u];
            num[cur]++;
        }
        for (int i = tot; i >= 2; i--) num[fail[i]] += num[i];
    }
}
```

