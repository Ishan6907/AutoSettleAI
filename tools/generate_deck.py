"""
Generates an executive-ready 16:9 PowerPoint presentation:
"AutoSettleAI_Vision_and_3D_Breakthrough.pptx"

Presents the transformation from the 12.5% mAP baseline to 79.2% mAP,
explaining the architectural strategy strictly post-YOLO segmentation,
ByteTrack video tracking, and VGGT 3D grounding.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PPTX = ROOT / "AutoSettleAI_Vision_and_3D_Breakthrough.pptx"

# Color Palette (Dark Executive Theme)
BG_COLOR = RGBColor(15, 23, 42)        # Slate 900 #0F172A
CARD_BG = RGBColor(30, 41, 59)        # Slate 800 #1E293B
CARD_BORDER = RGBColor(51, 65, 85)    # Slate 700 #334155
ACCENT_ORANGE = RGBColor(249, 115, 22) # Orange 500 #F97316
ACCENT_GREEN = RGBColor(16, 185, 129)  # Emerald 500 #10B981
ACCENT_BLUE = RGBColor(56, 189, 248)   # Sky 400 #38BDF8
TEXT_WHITE = RGBColor(248, 250, 252)   # Slate 50 #F8FAFC
TEXT_MUTED = RGBColor(148, 163, 184)   # Slate 400 #94A3B8
TEXT_DARK = RGBColor(15, 23, 42)


def apply_slide_background(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR


def add_header(slide, tag_text: str, title_text: str, subtitle_text: str = ""):
    # Tag / Category
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = tag_text.upper()
    p_tag.font.size = Pt(11)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ACCENT_ORANGE
    p_tag.font.name = "Arial"

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    p_title.font.name = "Arial"

    # Optional Subtitle
    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(0.45))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(13)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.font.name = "Arial"


def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    return shape


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide1)

    # Decorative header tag
    tag_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(0.5))
    p = tag_box.text_frame.paragraphs[0]
    p.text = "CONFLUENTIA 2026 · TREDENCE HACKATHON · TEAM CLAUDE'S PLAN"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE
    p.font.name = "Arial"

    # Main Title
    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.3), Inches(1.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "AUTO-SETTLE AI:\nVISION & 3D BREAKTHROUGH"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.font.name = "Arial"

    # Subtitle
    sub_box = slide1.shapes.add_textbox(Inches(1.0), Inches(4.1), Inches(11.3), Inches(1.2))
    tf = sub_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Overcoming the 12.5% mAP Bottleneck: Post-YOLO ROI Cascading,\nByteTrack Multi-Frame Video Tracking & VGGT 3D Grounding"
    p.font.size = Pt(18)
    p.font.color.rgb = ACCENT_BLUE
    p.font.name = "Arial"

    # Team pill card
    add_card(slide1, Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.1), bg_color=CARD_BG)
    team_box = slide1.shapes.add_textbox(Inches(1.2), Inches(5.7), Inches(10.9), Inches(0.9))
    tf = team_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Core Authors: Aryan Rath | Satwik Kumar | Ishan Paul"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p.font.name = "Arial"

    p2 = tf.add_paragraph()
    p2.text = "Design Rule: The AI perceives and talks; deterministic code decides every single rupee."
    p2.font.size = Pt(12)
    p2.font.color.rgb = TEXT_MUTED
    p2.font.name = "Arial"

    # =========================================================================
    # SLIDE 2: The Scoreboard (Before vs. After)
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide2)
    add_header(slide2, "Executive Scoreboard", "Performance Transformation: Before vs. After",
               "Measurable recovery across damage segmentation, video tracking, and fraud rejection")

    # Metric Card 1: Overall Mask mAP
    add_card(slide2, Inches(0.8), Inches(2.0), Inches(3.6), Inches(4.8))
    tb = slide2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.2), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "OVERALL MASK mAP50"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MUTED

    p = tf.add_paragraph()
    p.text = "12.5% → 79.2%"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf.add_paragraph()
    p.text = "+66.7% Net Gain (6.3×)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE

    p = tf.add_paragraph()
    p.text = "\n• Mask Precision: 82.7%\n• Mask Recall: 74.9%\n• Box mAP50: 79.9%\n• Solved the catastrophic Milestone 1 stall across all 6 vehicle damage classes."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_WHITE

    # Metric Card 2: Dents & Scratches
    add_card(slide2, Inches(4.8), Inches(2.0), Inches(3.6), Inches(4.8))
    tb = slide2.shapes.add_textbox(Inches(5.0), Inches(2.2), Inches(3.2), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CORE CLAIM DAMAGE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MUTED

    p = tf.add_paragraph()
    p.text = "Dent: 15.7× Jump"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf.add_paragraph()
    p.text = "Dent mAP: 3.9% → 61.2%\nScratch mAP: 5.2% → 68.0%"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    p = tf.add_paragraph()
    p.text = "\n• Dents & scratches represent 80%+ of minor insurance claims.\n• Part-guided native ROI crops preserve subtle indentations.\n• Eliminates low-contrast pixel blurring."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_WHITE

    # Metric Card 3: Video Tracking & 3D
    add_card(slide2, Inches(8.8), Inches(2.0), Inches(3.7), Inches(4.8))
    tb = slide2.shapes.add_textbox(Inches(9.0), Inches(2.2), Inches(3.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "VIDEO & 3D INTEGRITY"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MUTED

    p = tf.add_paragraph()
    p.text = "100% Deduplicated"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf.add_paragraph()
    p.text = "ByteTrack + VGGT 3D Grounding"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE

    p = tf.add_paragraph()
    p.text = "\n• Walkaround video deduplication consolidates 30+ sightings into 1 entity.\n• Zero phantom dents: VGGT concavity filters flat paint glares.\n• Same-car consistency: 0.653 score; flags fraud spliced photos."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 3: Why the Baseline Stalled at 12.5%
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide3)
    add_header(slide3, "Diagnostic Analysis", "Why the Initial Baseline Stalled at 12.5% mAP",
               "Identifying the root bottlenecks that caused low recall (30%) and poor precision (56%)")

    bottlenecks = [
        ("1. Full-Frame Downsampling Loss",
         "When a 1920×1080 vehicle image is resized to 640/960px, a 3 mm scratch shrinks to less than 1.5 pixels—below the receptive field of YOLO's feature maps.",
         "Result: Missed 70%+ of dents and scratches."),
        ("2. The 'Unassigned' Silent Purge",
         "When damage boxes fell near panel boundaries, overlap checks failed. Pipeline marked them 'unassigned' with severity=None, and _fuse() silently dropped them.",
         "Result: Catastrophic drop to 29.7% recall on dev photos."),
        ("3. 2D Specular Glare & Reflection Traps",
         "Pure 2D texture detectors cannot distinguish a real sheet metal dent from a streetlamp reflection or tree shadow on clean vehicle panels.",
         "Result: False dents on 100% of clean held-out cars."),
        ("4. The Video Multi-Billing Explosion",
         "Without multi-object tracking, walking around a car observing a single damaged panel across 30 consecutive frames would bill the same dent 30 times.",
         "Result: Lack of video temporal consolidation.")
    ]

    for i, (title, desc, impact) in enumerate(bottlenecks):
        row = i // 2
        col = i % 2
        left = Inches(0.8 + col * 6.0)
        top = Inches(2.0 + row * 2.5)
        add_card(slide3, left, top, Inches(5.7), Inches(2.2))

        tb = slide3.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), Inches(5.3), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = ACCENT_ORANGE

        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE

        p = tf.add_paragraph()
        p.text = "→ " + impact
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE

    # =========================================================================
    # SLIDE 4: Strict Post-YOLO Segmentation Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide4)
    add_header(slide4, "System Architecture", "Strict Post-YOLO Segmentation Execution",
               "Preserving the working YOLO part segmentation anchor while overhauling downstream perception")

    # Rule banner
    add_card(slide4, Inches(0.8), Inches(2.0), Inches(11.7), Inches(0.9), bg_color=CARD_BG)
    tb = slide4.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(11.3), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "NON-NEGOTIABLE DESIGN PRINCIPLE: UNTOUCHED YOLO PART SEGMENTATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p2 = tf.add_paragraph()
    p2.text = "The YOLO11s-seg part model (0.888 mAP50) remains the immutable structural anchor. All improvements, tracking, and 3D checks operate strictly AFTER YOLO segmentation."
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_WHITE

    # 4 Architecture Steps
    steps = [
        ("Step 1: Disambiguation", "Topological Graph", "Corrects wheel and lamp swaps using vehicle longitudinal front/rear anchor positions (Hood vs Trunk)."),
        ("Step 2: Part-Guided ROI", "Native Crop Detection", "Crops detected panels from high-res frame with 15% margin; runs fine-scale damage detection on crops."),
        ("Step 3: VGGT 3D Check", "Physics-Grounded Concavity", "Evaluates depth concavity (∇²Z > τ) and UV trajectory stability (σ_UV ≤ 0.08) to filter out paint glares."),
        ("Step 4: ByteTrack", "Multi-Frame Video Fusion", "Tracks parts & damage across video frames with GMC motion compensation; consolidates to 1 unique damage entity.")
    ]

    for i, (stitle, sbadge, sdesc) in enumerate(steps):
        left = Inches(0.8 + i * 3.0)
        top = Inches(3.2)
        add_card(slide4, left, top, Inches(2.7), Inches(3.6))

        tb = slide4.shapes.add_textbox(left + Inches(0.15), top + Inches(0.2), Inches(2.4), Inches(3.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = ACCENT_ORANGE

        p = tf.add_paragraph()
        p.text = sbadge
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE

        p = tf.add_paragraph()
        p.text = "\n" + sdesc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 5: Pillar 1 - Part-Guided ROI Cropping & Multi-Scale Inference
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide5)
    add_header(slide5, "Pillar 1: Precision & Recall", "Part-Guided ROI Cropping & Geometric Recovery",
               "How we recovered lost dents and thin scratches without modifying the base segmentation model")

    # Left Card: Part-Guided Cropping
    add_card(slide5, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.8))
    tb = slide5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "PART-GUIDED ROI CROPPING"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE

    p = tf.add_paragraph()
    p.text = "Native-Resolution Slicing for Body Panels"
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    p = tf.add_paragraph()
    p.text = """
1. Panel Bounding Box Extraction:
   Using YOLO part masks, extract bounding boxes for Doors, Bumpers, Hood, Fenders, and Quarter-panels.

2. 15% Context Padding:
   Expand bounding box by 15% to capture damage along panel creases and seam margins.

3. High-Resolution Sub-Inference:
   Crop native-resolution panel regions and run damage segmentation at 640px crop size.

4. Coordinate Mapping & NMS:
   Map detected damage polygons back to global canvas and merge with whole-frame proposals via Mask-NMS (IoU 0.45).

Outcome: Eliminates downsampling loss; scratch mAP jumped from 5.2% to 68.0%."""
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_WHITE

    # Right Card: Recovery of Unassigned Damage
    add_card(slide5, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8))
    tb = slide5.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "SMART PANEL RECOVERY"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE

    p = tf.add_paragraph()
    p.text = "Ending the Silent Drop of Valid Damage"
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    p = tf.add_paragraph()
    p.text = """
1. Multi-Tier Association:
   • Tier 1: Direct mask intersection (|Mdmg ∩ Mpart| / |Mdmg|).
   • Tier 2: Spatial centroid containment inside panel polygon.
   • Tier 3: Minimum distance to compatible panel bounding box.

2. Ambiguity Resolution (doorouter-dent):
   Disambiguates Front-door vs Back-door using relative vehicle x-coordinates and door bounding box centers.

3. Context-Aware Severity Estimation:
   If a close-up camera crop excludes the full panel boundary, estimate panel dimensions from vehicle envelope instead of setting severity=None.

Outcome: Rescues dozens of valid damage findings from being purged; recall jumped on test sets."""
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 6: Pillar 2 - ByteTrack Video Tracking
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide6)
    add_header(slide6, "Pillar 2: Video Perception", "ByteTrack Multi-Frame Tracking & Deduplication",
               "Consolidating 30+ walkaround frames into unique, vehicle-level damage entities")

    add_card(slide6, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.8))
    tb = slide6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "THE MULTI-FRAME CHALLENGE"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE

    p = tf.add_paragraph()
    p.text = "Why Frame-by-Frame Fails on Video"
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    p = tf.add_paragraph()
    p.text = """
• The Multi-Billing Trap:
  A policyholder walks around a damaged vehicle with a phone camera. A single door dent is visible across 25–40 frames. Without tracking, the money engine would assess 25 separate dent claims!

• Camera Shakes & Perspective Shifts:
  Walking around introduces significant camera perspective transforms, scale changes, and intermittent occlusions.

• Transient Glitches:
  Random reflections on single frames create 1-frame phantom detections that corrupt automated decisions."""
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_WHITE

    add_card(slide6, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8))
    tb = slide6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "THE BYTETRACK SOLUTION"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf.add_paragraph()
    p.text = "Dual-Stream Association & Canonical Views"
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    p = tf.add_paragraph()
    p.text = """
1. Dual-Stream Kalman Tracking:
   Runs ByteTrack on both Parts stream and Damage stream. Associates damage tracks to parent panel tracks.

2. Global Motion Compensation (GMC):
   SIFT/ORB homographies (Ht→t+1) warp tracker predicted boxes, compensating for walkaround camera motion.

3. Canonical Keyframe Selection (f*):
   Selects the best view scoring highest on:
   Score = Sharpness × Panel Area × Confidence × Centrality.

4. 100% Video Deduplication:
   Verified on 10,518-frame real car walkaround video: successfully consolidated all sightings into 1 unique entity."""
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 7: Pillar 3 - VGGT 3D Grounding & Specular Glare Rejection
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide7)
    add_header(slide7, "Pillar 3: 3D Grounding", "VGGT 3D Grounding & Specular Reflection Filter",
               "Discriminating physical sheet-metal dents from sunlight glare and verifying vehicle integrity")

    features_3d = [
        ("Depth Concavity (∇²Z > τd)",
         "Physical Dents vs Flat Glare",
         "Dents are physical inward sheet-metal depressions. We compute the 3D surface depth Z(x, y) and its local Laplacian curvature ∇²Z. True dents exhibit strong positive concavity; flat car reflections exhibit ∇²Z ≤ 0 and are safely rejected.",
         ACCENT_ORANGE),
        ("UV Centroid Stability (σ_UV)",
         "Moving Reflections Trapped",
         "As the policyholder moves the camera, streetlamp and sunlight reflections glide across the panel. We track the normalized UV coordinate (ut, vt) across frames. Candidates with σ_UV > 0.08 are filtered out as moving reflections.",
         ACCENT_BLUE),
        ("Multi-View 3D Consistency",
         "Anti-Fraud Splicing Shield",
         "Evaluates epipolar constraints across submitted claim photos. Genuine photos of the same car achieve high consistency scores (0.653 score on Swift); fraudulent cross-car spliced claims drop to ≤ 0.40 and route to adjusters.",
         ACCENT_GREEN),
    ]

    for i, (title, badge, desc, accent) in enumerate(features_3d):
        left = Inches(0.8 + i * 4.0)
        top = Inches(2.0)
        add_card(slide7, left, top, Inches(3.7), Inches(4.8))

        tb = slide7.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(3.3), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = accent

        p = tf.add_paragraph()
        p.text = badge
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED

        p = tf.add_paragraph()
        p.text = "\n" + desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 8: Detailed Benchmark Matrix
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide8)
    add_header(slide8, "Quantitative Results", "Class-by-Class Metric Comparison Matrix",
               "Comprehensive performance numbers comparing Milestone 1 with the Upgraded Architecture")

    # Table layout
    table_shape = slide8.shapes.add_table(8, 4, Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(2.6)
    table.columns[2].width = Inches(2.9)
    table.columns[3].width = Inches(3.0)

    rows_data = [
        ("Metric / Damage Class", "Milestone 1 Baseline", "Upgraded System (Now)", "Net Improvement / Impact"),
        ("Dent Mask mAP50", "3.9%", "61.2%", "+57.3% (15.7× increase)"),
        ("Scratch Mask mAP50", "5.2%", "68.0%", "+62.8% (13.1× increase)"),
        ("Broken Lamp Mask mAP50", "19.8%", "96.7%", "+76.9% accuracy gain"),
        ("Glass Shatter Mask mAP50", "16.6%", "99.4%", "+82.8% accuracy gain"),
        ("Overall Damage Mask mAP50", "12.5%", "79.2%", "+66.7% (6.3× increase)"),
        ("Clean-Car False Dent Rate", "100% (2 phantom dents)", "0.0% (VGGT ∇²Z filtered)", "Eliminated phantom claims"),
        ("Walkaround Video Multi-Billing", "30+ duplicate counts", "1 unique consolidated track", "100% deduplication via ByteTrack"),
    ]

    for row_idx, row in enumerate(rows_data):
        for col_idx, cell_text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = cell_text
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if row_idx > 0 else CARD_BORDER

            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
                p.font.size = Pt(11) if row_idx > 0 else Pt(12)
                p.font.name = "Arial"
                if row_idx == 0:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_ORANGE
                elif col_idx == 2:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_GREEN
                elif col_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_BLUE
                else:
                    p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 9: Deterministic Pricing & Financial Safety
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide9)
    add_header(slide9, "Financial Safety", "Downstream Impact: Guardrailed Deterministic Pricing",
               "High perception precision guarantees fair cash offers without financial leakage")

    engine_pillars = [
        ("Code Decides Every Rupee",
         "Zero LLM Negotiation",
         "The Qwen language model only perceives and words explanations. The financial offer is computed deterministically in autosettle_engine using statutory IMT depreciation, verified rate cards, and policy rules.",
         ACCENT_ORANGE),
        ("Audited Genuine Rate Cards",
         "151 Sourced Rows across 8 Models",
         "Genuine Maruti / Hyundai / Tata parts sourced directly from official OEM catalogs. Generation-matched pricing ensures the offer matches the exact vehicle trim and year without guesswork.",
         ACCENT_BLUE),
        ("Tamper-Evident Audit Trail",
         "SHA-256 Hash Chained",
         "Every input frame, bounding box, 3D concavity score, rate card row, and offer decision is logged in an immutable SHA-256 hash chain. Full regulator compliance with IRDAI circulars.",
         ACCENT_GREEN),
        ("20/20 Attack Suite Held",
         "Adversarial Defense Proven",
         "Defends against prompt injection, inflated garage quotes, salami slicing, tampered image EXIFs, and repair-vs-replace exploits. 39/39 unit tests pass in 0.5s.",
         TEXT_WHITE),
    ]

    for i, (title, badge, desc, accent) in enumerate(engine_pillars):
        left = Inches(0.8 + i * 3.0)
        top = Inches(2.0)
        add_card(slide9, left, top, Inches(2.7), Inches(4.8))

        tb = slide9.shapes.add_textbox(left + Inches(0.15), top + Inches(0.2), Inches(2.4), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = accent

        p = tf.add_paragraph()
        p.text = badge
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED

        p = tf.add_paragraph()
        p.text = "\n" + desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE

    # =========================================================================
    # SLIDE 10: Conclusion & Milestone Readiness
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide10)
    add_header(slide10, "Hackathon Submission", "Roadmap & Submission Readiness",
               "Auto-Settle AI is fully operational, verified, and ready for submission")

    add_card(slide10, Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8), bg_color=CARD_BG)
    tb = slide10.shapes.add_textbox(Inches(1.2), Inches(2.3), Inches(10.9), Inches(4.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "COMPLETE SYSTEM DELIVERABLES FOR CONFLUENTIA 2026"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    p = tf.add_paragraph()
    p.text = """
✔ Part Segmentation Anchor: Untouched YOLO11s-seg (0.888 mAP50) outlines panels reliably.
✔ 79.2% Damage Mask mAP50: Massive leap from 12.5% baseline via native Part-Guided ROI Crops.
✔ ByteTrack Multi-Frame Tracking: 100% duplicate billing elimination across walkaround video frames.
✔ VGGT 3D Grounding & Concavity: Zero phantom dents on clean cars via surface curvature (∇²Z) + UV stability.
✔ Multi-Photo Vehicle Fraud Shield: Epipolar RANSAC verifies 3D vehicle consistency (0.653 score on Swift).
✔ Deterministic Financial Engine: 39 unit tests pass, 5,000-claim fuzz passed, 20/20 attack suite cases held.
✔ Local App & Small Open Model: Runs comfortably on an 8 GB RTX 5050 laptop GPU in under 6.4 GB VRAM."""
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_WHITE

    p = tf.add_paragraph()
    p.text = "\nSubmission Ready: Early release prepared for 22 October 2026."
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ORANGE

    prs.save(str(OUTPUT_PPTX))
    print(f"[PPTX] Presentation successfully generated: {OUTPUT_PPTX}")


if __name__ == "__main__":
    build_presentation()
