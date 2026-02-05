# Jie Yang's Homepage

Personal academic website built with Jekyll.

## Project Structure

```
├── _blogs/          # 博客文章 (Markdown)
├── _publications/   # 发表论文
├── _pages_/         # 页面模板
├── files/           # 静态文件（图片等）
│   └── blog_assets/ # 博客图片
└── blogs/
    └── files -> ../files  # 符号链接
```

## Blog Usage

### Add a new blog

在 `_blogs/` 中创建 `.md` 文件：

```yaml
---
title: "Your Blog Title"
date: 2026-01-01
---

Blog content here...
```

### Blog ordering

博客按文件名排序，使用数字前缀控制顺序：

```
0-first-post.md
1-second-post.md
2-third-post.md
```

### Blog images

图片放在 `files/blog_assets/`，使用相对路径引用：

```markdown
![image](../files/blog_assets/your-image.png)
```

### Why symbolic link?

本地文件结构和线上 URL 结构不一致：

| 环境 | 路径 |
|------|------|
| 本地文件 | `_blogs/post.md` (一层) |
| 线上 URL | `/blogs/post/` (两层) |

相对路径 `../files/...` 在线上会解析到 `/blogs/files/...`。

符号链接 `blogs/files -> ../files` 让这个路径能正确指向 `/files/`。

**If the symlink is missing, recreate it:**

```bash
mkdir -p blogs
ln -s ../files blogs/files
```

## Local Development

```bash
bundle install
bundle exec jekyll serve
```

Visit `http://localhost:4000`
