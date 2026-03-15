//! PDF to image rendering with platform-specific backends.
//!
//! - macOS: CoreGraphics (CGPDFDocument) — zero extra dependencies
//! - Other: pdfium-render (bundles PDFium C++ library)

use crate::error::Error;

/// A rendered PDF page as an RGB image.
pub struct RenderedPage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGB bytes, row-major
}

/// Render all pages of a PDF to RGB images.
///
/// `dpi` controls the resolution (default recommendation: 200 for OCR).
pub fn render_pdf_pages(pdf_bytes: &[u8], dpi: u32) -> Result<Vec<RenderedPage>, Error> {
    render_pdf_pages_impl(pdf_bytes, dpi)
}

/// Check if a byte slice looks like a PDF (starts with `%PDF`).
pub fn is_pdf(data: &[u8]) -> bool {
    data.len() >= 4 && &data[..4] == b"%PDF"
}

// ============================================================================
// macOS: CoreGraphics backend
// ============================================================================

#[cfg(target_os = "macos")]
fn render_pdf_pages_impl(pdf_bytes: &[u8], dpi: u32) -> Result<Vec<RenderedPage>, Error> {
    use std::ffi::c_void;

    // CoreGraphics FFI
    #[allow(non_camel_case_types)]
    type CGPDFDocumentRef = *const c_void;
    #[allow(non_camel_case_types)]
    type CGPDFPageRef = *const c_void;
    #[allow(non_camel_case_types)]
    type CGDataProviderRef = *const c_void;
    #[allow(non_camel_case_types)]
    type CGContextRef = *mut c_void;
    #[allow(non_camel_case_types)]
    type CGColorSpaceRef = *const c_void;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct CGRect {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct CGAffineTransform {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        tx: f64,
        ty: f64,
    }

    // CGPDFBox enum value for media box
    const K_CG_PDF_MEDIA_BOX: i32 = 0;
    // Bitmap info: premultiplied last (RGBA) with byte order default
    const K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST: u32 = 1;

    extern "C" {
        fn CGDataProviderCreateWithData(
            info: *mut c_void,
            data: *const c_void,
            size: usize,
            release_data: *const c_void,
        ) -> CGDataProviderRef;
        fn CGDataProviderRelease(provider: CGDataProviderRef);
        fn CGPDFDocumentCreateWithProvider(provider: CGDataProviderRef) -> CGPDFDocumentRef;
        fn CGPDFDocumentRelease(document: CGPDFDocumentRef);
        fn CGPDFDocumentGetNumberOfPages(document: CGPDFDocumentRef) -> usize;
        fn CGPDFDocumentGetPage(document: CGPDFDocumentRef, page_number: usize) -> CGPDFPageRef;
        fn CGPDFPageGetBoxRect(page: CGPDFPageRef, box_type: i32) -> CGRect;
        fn CGColorSpaceCreateDeviceRGB() -> CGColorSpaceRef;
        fn CGColorSpaceRelease(space: CGColorSpaceRef);
        fn CGBitmapContextCreate(
            data: *mut c_void,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: CGColorSpaceRef,
            bitmap_info: u32,
        ) -> CGContextRef;
        fn CGContextRelease(context: CGContextRef);
        fn CGContextSetRGBFillColor(
            context: CGContextRef,
            r: f64,
            g: f64,
            b: f64,
            a: f64,
        );
        fn CGContextFillRect(context: CGContextRef, rect: CGRect);
        fn CGContextScaleCTM(context: CGContextRef, sx: f64, sy: f64);
        fn CGContextDrawPDFPage(context: CGContextRef, page: CGPDFPageRef);
    }

    let scale = dpi as f64 / 72.0;

    // Create data provider from bytes
    let provider = unsafe {
        CGDataProviderCreateWithData(
            std::ptr::null_mut(),
            pdf_bytes.as_ptr() as *const c_void,
            pdf_bytes.len(),
            std::ptr::null(),
        )
    };
    if provider.is_null() {
        return Err(Error::Model("Failed to create CGDataProvider".into()));
    }

    let document = unsafe { CGPDFDocumentCreateWithProvider(provider) };
    if document.is_null() {
        unsafe { CGDataProviderRelease(provider) };
        return Err(Error::Model("Failed to parse PDF document".into()));
    }

    let num_pages = unsafe { CGPDFDocumentGetNumberOfPages(document) };
    let mut pages = Vec::with_capacity(num_pages);

    let color_space = unsafe { CGColorSpaceCreateDeviceRGB() };

    for page_idx in 1..=num_pages {
        let page = unsafe { CGPDFDocumentGetPage(document, page_idx) };
        if page.is_null() {
            continue;
        }

        let media_box = unsafe { CGPDFPageGetBoxRect(page, K_CG_PDF_MEDIA_BOX) };
        let pixel_w = (media_box.width * scale).ceil() as usize;
        let pixel_h = (media_box.height * scale).ceil() as usize;

        // RGBA buffer
        let bytes_per_row = pixel_w * 4;
        let mut buffer = vec![0u8; pixel_h * bytes_per_row];

        let context = unsafe {
            CGBitmapContextCreate(
                buffer.as_mut_ptr() as *mut c_void,
                pixel_w,
                pixel_h,
                8,
                bytes_per_row,
                color_space,
                K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST,
            )
        };
        if context.is_null() {
            continue;
        }

        // White background
        unsafe {
            CGContextSetRGBFillColor(context, 1.0, 1.0, 1.0, 1.0);
            CGContextFillRect(
                context,
                CGRect {
                    x: 0.0,
                    y: 0.0,
                    width: pixel_w as f64,
                    height: pixel_h as f64,
                },
            );
        }

        // Scale and draw
        unsafe {
            CGContextScaleCTM(context, scale, scale);
            CGContextDrawPDFPage(context, page);
            CGContextRelease(context);
        }

        // Convert RGBA to RGB
        let mut rgb_data = Vec::with_capacity(pixel_w * pixel_h * 3);
        for pixel in buffer.chunks_exact(4) {
            rgb_data.push(pixel[0]);
            rgb_data.push(pixel[1]);
            rgb_data.push(pixel[2]);
        }

        pages.push(RenderedPage {
            width: pixel_w as u32,
            height: pixel_h as u32,
            data: rgb_data,
        });
    }

    unsafe {
        CGColorSpaceRelease(color_space);
        CGPDFDocumentRelease(document);
        CGDataProviderRelease(provider);
    }

    if pages.is_empty() {
        return Err(Error::Model("PDF has no renderable pages".into()));
    }

    Ok(pages)
}

// ============================================================================
// Non-macOS: pdfium-render backend
// ============================================================================

#[cfg(not(target_os = "macos"))]
fn render_pdf_pages_impl(pdf_bytes: &[u8], dpi: u32) -> Result<Vec<RenderedPage>, Error> {
    use pdfium_render::prelude::*;

    let pdfium = Pdfium::default();
    let document = pdfium
        .load_pdf_from_byte_slice(pdf_bytes, None)
        .map_err(|e| Error::Model(format!("Failed to parse PDF: {}", e)))?;

    let scale = dpi as f32 / 72.0;
    let mut pages = Vec::new();

    for page in document.pages().iter() {
        let width = page.width();
        let height = page.height();
        let pixel_w = (width.value * scale) as u32;
        let pixel_h = (height.value * scale) as u32;

        let bitmap = page
            .render_with_config(
                &PdfRenderConfig::new()
                    .set_target_width(pixel_w as i32)
                    .set_target_height(pixel_h as i32),
            )
            .map_err(|e| Error::Model(format!("Failed to render PDF page: {}", e)))?;

        let img = bitmap.as_image();
        let rgb = img.to_rgb8();
        let data = rgb.into_raw();

        pages.push(RenderedPage {
            width: pixel_w,
            height: pixel_h,
            data,
        });
    }

    if pages.is_empty() {
        return Err(Error::Model("PDF has no renderable pages".into()));
    }

    Ok(pages)
}
