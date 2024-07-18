## 数学

### 试除法判质数

```cpp
bool is_prime(int x)
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}
```

### 试除法分解质因数

```cpp
void divide(int x)
{
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int s = 0;
            while (x % i == 0) x /= i, s ++ ;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}
```

### 线性筛法

```cpp
int primes[N], cnt;     // primes[]存储所有素数
bool st[N];         // st[x]存储x是否被筛掉

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; j < cnt && primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}
```

### 欧几里得算法求最大公因数

```cpp
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}
```

### 裴蜀定理

如果 $a,b$ 均为整数，一定存在整数 $x,y$ 使得 $ax+by = gcd(a,b)$ 成立。

推论：对于方程 $ax+by=c$，如果 $gcd(a,b)\mid c$​，则方程一定有解，反之一定无解。

### 扩展欧几里得算法

求得的是 $ax+by=gcd(a,b)$ 的一组特解，该方程的通解可以表示为 $\begin{cases}x'=x+k\frac{b}{gcd(a,b)}\\y'=y-k\frac{a}{gcd(a,b)}\end{cases}(k\in Z)$ .​

```cpp
int exgcd(int a, int b, int &x, int &y) {
    if(b == 0) {
        x = 1, y = 0;
        return a;
    }
    int gcd = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return gcd;
}
```

### 快速幂

```cpp
int qpow(int x, int k, int Mod) {
    int res = 1;
    while(k) {
        if(k & 1)
            res = 1ll * res * x % Mod;
        x = 1ll * x * x % Mod;
        k >>= 1;
    }
    return res;
}
```

### 中国剩余定理

```cpp
ll CRT(int n, ll a[], ll b[]) // a为模数，b为余数
{
    ll ans = 0;
    ll s = 1;
    for (int i = 1; i <= n; i++)
    {
        s *= a[i];
    }
    for (int i = 1; i <= n; i++)
    {
        ll m = s / a[i];
        ll x, y;
        exgcd(m, a[i], x, y);                     // x为m在模a[i]下的逆元
        ans = (ans + (m * x * b[i]) % s + s) % s; // m*x不要对a[i]取模
    }
    return ans;
}
```

### 求乘法逆元

1. 当mod为质数时

   $ax\equiv 1 \pmod b$ 由费马小定理有 $ax\equiv a^{b-1}\pmod b$

   $\therefore x\equiv a^{b-2} \pmod b$.

   快速幂求 $a^{b-2}$ 即可

   ```cpp
   long long qpow(long long a, int b)
   {
       long long res = 1;
       for (int i = 0; (1 << i) <= b; i++)
       {
           if ((b >> i) & 1)
               res = (res * a) % mod;
           a = (a * a) % mod;
       }
       return res;
   }
   ```

2. 扩展欧几里得方法（要求 $\gcd(a,b)=1$ ）

   等价于求 $ax\equiv 1\pmod p$ 的解，可以写为 $ax+pk = 1$，求解 $x,k$ 即可。

   ```cpp
   void exgcd(ll a, ll b, ll &x, ll &y)
   {
       if (b == 0)
       {
           x = 1;
           y = 0;
           return;
       }
       exgcd(b, a % b, x, y);
       ll tmp = x;
       x = y;
       y = tmp - a / b * y;
   }
   ```

### FFT

递归版

```cpp
void FFT(complex<double> *A, int limit, int op)
{
    if (limit == 1)
        return;
    complex<double> A1[limit / 2], A2[limit / 2];
    for (int i = 0; i < limit / 2; i++)
    {
        A1[i] = A[i * 2], A2[i] = A[i * 2 + 1];
    }
    FFT(A1, limit / 2, op), FFT(A2, limit / 2, op);
    complex<double> w1  ({cos(2 * pi / limit), sin(2 * pi / limit) * op});
    complex<double> wk({1, 0});
    for (int i = 0; i < limit / 2; i++)
    {
        A[i] = A1[i] + A2[i] * wk;
        A[i + limit / 2] = A1[i] - A2[i] * wk;
        wk = wk * w1;
    }
}
```

迭代版

```cpp
void change(complex<double> *A, int len)
{
    for (int i = 0; i < len; i++)
        R[i] = R[i / 2] / 2 + ((i & 1) ? len / 2 : 0);
    for (int i = 0; i < len; i++)
        if (i < R[i]) swap(A[i], A[R[i]]);
}

void FFT(complex<double> *A, int limit, int op)
{
    change(A, limit);
    for (int k = 2; k <= limit; k <<= 1)
    {
        complex<double> w1  ({cos(2 * pi / k), sin(2 * pi / k) * op});
        for (int i = 0; i < limit; i += k)
        {
            complex<double> wk({1, 0});
            for (int j = 0; j < k / 2; j++)
            {
                complex<double> x = A[i + j], y = A[i + j + k / 2] * wk;
                A[i + j] = x + y;
                A[i + j + k / 2] = x - y;
                wk = wk * w1;
            }
        }
    }
}
```

