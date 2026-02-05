---
layout: archive
title: "Blogs"
permalink: /blogs/
author_profile: true
---
{% include base_path %}

{% assign sorted_blogs = site.blogs | sort: 'path' %}
{% for post in sorted_blogs %}
  {% include archive-single.html %}
{% endfor %}
