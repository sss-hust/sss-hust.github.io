// scripts/protect-math.js

hexo.extend.filter.register('marked:extensions', function (extensions) {
  // 1. 处理块级公式 $$...$$
  extensions.push({
    name: 'blockMath',
    level: 'block',
    start(src) {
      return src.indexOf('$$');
    },
    tokenizer(src, tokens) {
      // 关键修改：把 ^ {0,3} 改为 ^\s*，允许任意开头的空白字符（包括换行）
      // 并且优化了中间内容的匹配逻辑
      const rule = /^\s*\$\$([\s\S]*?)\$\$/;
      const match = rule.exec(src);
      if (match) {
        return {
          type: 'blockMath',
          raw: match[0],
          text: match[1].trim() // 去掉前后多余空白，保留公式内容
        };
      }
    },
    renderer(token) {
      // 输出时补回 $$，确保 hexo-filter-mathjax 能识别
      return `$$${token.text}$$\n`;
    }
  });

  // 2. 处理行内公式 $...$
  extensions.push({
    name: 'inlineMath',
    level: 'inline',
    start(src) {
      return src.indexOf('$');
    },
    tokenizer(src, tokens) {
      // 匹配 $...$，排除 \$ 转义，允许公式包含空格但不能有换行
      // 注意：marked 的 inline tokenizer 会从当前位置开始匹配，所以需要 ^
      // 但我们需要确保 $ 前面不是转义符 \
      const rule = /^\$([^\$\n]+?)\$/;
      const match = rule.exec(src);
      if (match) {
        return {
          type: 'inlineMath',
          raw: match[0],
          text: match[1]
        };
      }
    },
    renderer(token) {
      return `$${token.text}$`;
    }
  });
});