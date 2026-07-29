"""Small in-memory PDF builders shared by document tests."""


def _pdf_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_text_pdf(*page_texts: str, title: str | None = None) -> bytes:
    """Build a minimal text-layer PDF for parser and fetcher tests."""

    page_count = len(page_texts)
    font_id = 3 + page_count
    content_start_id = font_id + 1
    info_id = content_start_id + page_count if title is not None else None
    page_ids = list(range(3, 3 + page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
            f"/Count {page_count} >>"
        ).encode(),
    ]

    for index, page_id in enumerate(page_ids):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_start_id + index} 0 R >>"
            ).encode()
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_text in page_texts:
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({_pdf_string(page_text)}) Tj ET"
        ).encode()
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

    if title is not None:
        objects.append(f"<< /Title ({_pdf_string(title)}) >>".encode())

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())

    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R"
    if info_id is not None:
        trailer += f" /Info {info_id} 0 R"
    trailer += f" >>\nstartxref\n{xref_offset}\n%%EOF\n"
    output.extend(trailer.encode())
    return bytes(output)
