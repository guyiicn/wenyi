"""CSS-based emphasis detection for span elements.

Reliably identifies span elements with font-style:italic or font-weight:bold
from external CSS or inline styles. Does not guess from class names alone.
"""

from __future__ import annotations

import re
from typing import Any

from bs4 import Tag


def _parse_inline_style(style_str: str) -> dict[str, str]:
    """Parse inline style attribute into property: value dict."""
    result = {}
    if not isinstance(style_str, str):
        return result

    for declaration in style_str.split(";"):
        declaration = declaration.strip()
        if ":" in declaration:
            prop, value = declaration.split(":", 1)
            result[prop.strip().lower()] = value.strip().lower()

    return result


def _parse_css_rules(css_text: str) -> dict[str, dict[str, str]]:
    """Parse CSS text into selector: {property: value} mapping.

    Handles basic CSS: .class, #id, tag, tag.class selectors.
    Returns normalized properties (lowercase keys/values).
    """
    rules: dict[str, dict[str, str]] = {}

    # Remove comments
    css_text = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)

    # Match rule blocks: selector { declarations }
    rule_pattern = r"([^{]+)\{([^}]+)\}"

    for match in re.finditer(rule_pattern, css_text):
        selector_group = match.group(1).strip()
        declarations = match.group(2).strip()

        # Parse declarations
        props = {}
        for decl in declarations.split(";"):
            decl = decl.strip()
            if ":" in decl:
                prop, value = decl.split(":", 1)
                props[prop.strip().lower()] = value.strip().lower()

        # Handle multiple selectors (comma-separated)
        for selector in selector_group.split(","):
            selector = selector.strip()
            if selector:
                rules[selector] = props

    return rules


def _matches_selector(element: Tag, selector: str) -> bool:
    """Check if element matches a simple CSS selector.

    Supports: .class, #id, tag, tag.class, tag#id
    Does not support complex selectors (descendant, child, etc.)
    """
    selector = selector.strip()

    # Class selector: .classname
    if selector.startswith("."):
        class_name = selector[1:]
        element_classes = element.get("class", [])
        if isinstance(element_classes, list):
            return class_name in element_classes
        return False

    # ID selector: #idname
    if selector.startswith("#"):
        id_name = selector[1:]
        return element.get("id") == id_name

    # Tag with class: tag.classname
    if "." in selector:
        tag_part, class_part = selector.split(".", 1)
        if element.name != tag_part:
            return False
        element_classes = element.get("class", [])
        if isinstance(element_classes, list):
            return class_part in element_classes
        return False

    # Tag with ID: tag#idname
    if "#" in selector:
        tag_part, id_part = selector.split("#", 1)
        if element.name != tag_part:
            return False
        return element.get("id") == id_part

    # Tag selector
    return element.name == selector


def has_italic_style(element: Tag, css_rules: dict[str, dict[str, str]] | None = None) -> bool:
    """Check if element has font-style: italic from inline style or CSS rules."""
    # Check inline style
    inline_style = element.get("style")
    if inline_style:
        props = _parse_inline_style(inline_style)
        font_style = props.get("font-style", "")
        if font_style == "italic":
            return True

    # Check CSS rules
    if css_rules:
        for selector, props in css_rules.items():
            if _matches_selector(element, selector):
                font_style = props.get("font-style", "")
                if font_style == "italic":
                    return True

    return False


def has_bold_style(element: Tag, css_rules: dict[str, dict[str, str]] | None = None) -> bool:
    """Check if element has font-weight: bold/700+ from inline style or CSS rules."""
    # Check inline style
    inline_style = element.get("style")
    if inline_style:
        props = _parse_inline_style(inline_style)
        font_weight = props.get("font-weight", "")
        if font_weight == "bold" or font_weight == "bolder":
            return True
        # Numeric weight >= 700 is bold
        if font_weight.isdigit() and int(font_weight) >= 700:
            return True

    # Check CSS rules
    if css_rules:
        for selector, props in css_rules.items():
            if _matches_selector(element, selector):
                font_weight = props.get("font-weight", "")
                if font_weight == "bold" or font_weight == "bolder":
                    return True
                if font_weight.isdigit() and int(font_weight) >= 700:
                    return True

    return False


def extract_css_rules_from_epub(
    soup: Any, *, epub_zip=None, resource_href: str | None = None
) -> dict[str, dict[str, str]]:
    """Extract CSS rules from <style> tags and <link rel="stylesheet"> in EPUB HTML.

    Returns merged rules from all style blocks and linked CSS files.
    
    Args:
        soup: BeautifulSoup parsed HTML
        epub_zip: Optional ZipFile object to read external CSS files
        resource_href: Current XHTML resource path (for resolving relative CSS links)
    """
    merged_rules: dict[str, dict[str, str]] = {}

    # Extract from inline <style> tags
    for style_tag in soup.find_all("style"):
        css_text = style_tag.get_text()
        rules = _parse_css_rules(css_text)

        # Merge rules (later rules override earlier ones)
        for selector, props in rules.items():
            if selector in merged_rules:
                merged_rules[selector].update(props)
            else:
                merged_rules[selector] = props.copy()

    # Extract from external <link rel="stylesheet"> files
    if epub_zip and resource_href:
        import os
        
        for link_tag in soup.find_all("link", rel="stylesheet"):
            css_href = link_tag.get("href")
            if not css_href:
                continue
            
            # Resolve relative path
            if css_href.startswith("/"):
                css_path = css_href.lstrip("/")
            else:
                # Relative to current XHTML resource
                resource_dir = os.path.dirname(resource_href)
                css_path = os.path.normpath(os.path.join(resource_dir, css_href)).replace("\\", "/")
            
            # Read CSS file from EPUB
            try:
                if css_path in epub_zip.namelist():
                    css_content = epub_zip.read(css_path).decode("utf-8", errors="ignore")
                    rules = _parse_css_rules(css_content)
                    
                    # Merge rules
                    for selector, props in rules.items():
                        if selector in merged_rules:
                            merged_rules[selector].update(props)
                        else:
                            merged_rules[selector] = props.copy()
            except Exception:
                # Ignore CSS file read errors
                pass

    return merged_rules
