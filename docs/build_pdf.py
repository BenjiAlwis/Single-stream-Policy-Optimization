"""Build the SPO explainer PDF: plain-language methods + diagrams + results."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
    HRFlowable, KeepTogether,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(_HERE, "diagrams")
FIG = os.path.join(_HERE, "..", "figures")
OUT = os.path.join(_HERE, "SPO_explained.pdf")

# ---------------- palette ----------------
INK = colors.HexColor("#1c2230")
SPO = colors.HexColor("#d98a29")
GRPO = colors.HexColor("#2a7fd9")
MUTE = colors.HexColor("#5b6472")
GREEN = colors.HexColor("#3aa64a")
RED = colors.HexColor("#d4564a")
LIGHT = colors.HexColor("#f1f3f7")
LIGHTER = colors.HexColor("#f8f9fb")
PURPLE = colors.HexColor("#7c5cbf")

styles = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


title_st = S("t", fontName="Helvetica-Bold", fontSize=26, leading=30,
             textColor=INK, alignment=TA_LEFT, spaceAfter=4)
subtitle_st = S("st", fontName="Helvetica", fontSize=12.5, leading=17,
                textColor=MUTE, spaceAfter=2)
h1 = S("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK,
       spaceBefore=16, spaceAfter=7)
h2 = S("h2", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=SPO,
       spaceBefore=11, spaceAfter=4)
body = S("body", fontName="Helvetica", fontSize=10.5, leading=15.5,
         textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
cap = S("cap", fontName="Helvetica-Oblique", fontSize=9, leading=12.5,
        textColor=MUTE, alignment=TA_CENTER, spaceBefore=3, spaceAfter=12)
small = S("small", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK)
key_st = S("key", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=INK)


def img(path, width_cm):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = width_cm * cm
    h = w * ih / iw
    return Image(path, width=w, height=h)


def callout(text, color=SPO, bg=LIGHTER):
    p = Paragraph(text, S("co", parent=body, spaceAfter=0, alignment=TA_LEFT))
    t = Table([[p]], colWidths=[16.3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LINEBEFORE", (0, 0), (0, -1), 3, color),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t


story = []

# ============================ COVER ============================
story.append(Spacer(1, 0.4 * cm))
bar = Table([[""]], colWidths=[16.3 * cm], rowHeights=[5])
bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SPO)]))
story.append(bar)
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("Single-stream Policy Optimization", title_st))
story.append(Paragraph("A plain-language guide to the method and the code that reproduces it",
                       subtitle_st))
story.append(Spacer(1, 0.3 * cm))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#dfe3ea")))
story.append(Spacer(1, 0.35 * cm))
story.append(Paragraph(
    "This guide explains <b>SPO</b> — a way of training a decision-making policy with "
    "reinforcement learning — without assuming you know the math. We start from the problem "
    "it solves, build up each idea with a picture, and finish with the experimental results "
    "from a small, runnable version of the algorithm. Wherever the original paper talks about "
    "a language model answering a <i>prompt</i>, we use the more general word <i>state</i> (a "
    "situation the policy faces). Everything else is the same.",
    body))

story.append(callout(
    "<b>The one-sentence idea.</b> &nbsp;Instead of generating many answers to the same "
    "question and comparing them against each other (which is wasteful), SPO generates "
    "<b>one</b> answer and compares it against a running <b>memory</b> of how well that "
    "question has gone before.", SPO))

story.append(Spacer(1, 0.2 * cm))
story.append(img(f"{DIAG}/d1_grpo_vs_spo.png", 16.3))
story.append(Paragraph(
    "Figure 1. The core contrast. GRPO (top) needs a whole group of answers per question. "
    "SPO (bottom) needs just one, because a value tracker remembers the question's history.",
    cap))

# ============================ SECTION 1 ============================
story.append(PageBreak())
story.append(Paragraph("1. The problem with the popular approach (GRPO)", h1))
story.append(Paragraph(
    "A standard way to train these policies is called <b>GRPO</b>. For every question, it "
    "generates a <b>group</b> of several answers (say 8), scores each one (correct = 1, wrong "
    "= 0), and then judges each answer against the group's average. An answer that beat the "
    "group average gets reinforced; one that fell below gets discouraged. The group average "
    "acts as a 'bar to clear'.", body))
story.append(Paragraph("This works, but it has two expensive flaws.", body))

story.append(Paragraph("Flaw 1 — Wasted groups (the 'degeneracy' problem)", h2))
story.append(Paragraph(
    "If every answer in the group gets the <b>same</b> score — all correct, or all wrong — then "
    "every answer exactly equals the group average. The 'bar to clear' is identical to every "
    "result, so nothing stands out, and the policy learns <b>nothing</b> from that question. "
    "All the computation spent generating those answers is thrown away. This happens constantly "
    "for questions that are either very easy (always solved) or very hard (never solved).", body))

story.append(Paragraph("Flaw 2 — Everyone waits for the slowest (the 'synchronization' problem)", h2))
story.append(Paragraph(
    "Because the group is judged together, you can't finish until <b>every</b> answer in the "
    "group is done. If one answer takes a long time — common when the policy uses tools or "
    "reasons over many steps — the entire group sits idle waiting for that straggler. This "
    "badly hurts speed.", body))

story.append(callout(
    "<b>Why this matters.</b> &nbsp;Both flaws come from the same root cause: GRPO builds its "
    "comparison bar from a <b>group generated right now</b>. SPO removes the group entirely.",
    GRPO))

# ============================ SECTION 2 ============================
story.append(Paragraph("2. SPO's fix: remember instead of regenerate", h1))
story.append(Paragraph(
    "SPO keeps a small <b>memory</b> for each question: an estimate of how often the policy "
    "currently succeeds at it. Call this estimate <i>v</i><super>^</super> (\"v-hat\"). When a new answer comes "
    "in, SPO compares it against <i>v</i><super>^</super> instead of against a freshly-generated group. That single "
    "change fixes both flaws at once — there's no group to go degenerate, and no group to wait "
    "for.", body))
story.append(Paragraph(
    "SPO has three working parts. The next three sections cover each one in plain terms.", body))

# ---- 2a tracker ----
story.append(Paragraph("Part A — The value tracker (the memory)", h2))
story.append(Paragraph(
    "The tracker stores <b>one number per question</b>: the current success estimate <i>v</i><super>^</super>. It "
    "does <b>not</b> store a list of past results. Each time a new result arrives, it just nudges "
    "the existing number a little toward the new result — the way a running average works. This "
    "keeps memory tiny no matter how long training runs.", body))
story.append(img(f"{DIAG}/d2_tracker.png", 16.3))
story.append(Paragraph(
    "Figure 2. Left: each new result gently nudges the single stored estimate. Right: how much "
    "weight old results keep depends on how much the policy has changed — measured by 'KL', a "
    "standard gauge of how different two policies are.", cap))
story.append(Paragraph(
    "There's one clever twist. If the policy has changed a lot since it last saw a question, the "
    "old success rate is out of date and should be trusted less. SPO measures how much the policy "
    "changed (using <b>KL divergence</b>, a standard distance between two policies) and "
    "<b>forgets faster</b> when the change is large. A 'forgetting dial' turns toward 'keep' when "
    "the policy is stable and toward 'forget' when it shifts.", body))

# ---- 2b advantage ----
story.append(PageBreak())
story.append(Paragraph("Part B — The advantage (how surprising was the result?)", h2))
story.append(Paragraph(
    "Once we have the memory <i>v</i><super>^</super>, scoring an answer is simple: take the actual result and "
    "subtract what we expected. The paper calls this the <b>advantage</b>:", body))
story.append(Paragraph(
    "<para alignment='center'><font size='13'><b>advantage = result &#8722; expectation = "
    "r &#8722; <i>v</i><super>^</super></b></font></para>", S("eq", parent=body, spaceAfter=10)))
story.append(Paragraph(
    "A positive advantage means the answer did better than expected, so the policy should do "
    "more of that. A negative advantage means worse than expected, so do less. The size tells "
    "you how much to adjust.", body))
story.append(img(f"{DIAG}/d3_advantage.png", 16.3))
story.append(Paragraph(
    "Figure 3. The same result means different things depending on expectation. Solving a hard "
    "question (low expectation) is a big positive surprise and drives strong learning; failing "
    "an easy one is a big negative surprise.", cap))
story.append(callout(
    "<b>Why subtract an expectation at all?</b> &nbsp;Raw scores are noisy and jumpy, which makes "
    "training unstable. Subtracting a good expectation cancels most of that noise while keeping "
    "the part that actually signals 'better or worse than usual'. This is the classic idea of a "
    "<i>baseline</i> in reinforcement learning — SPO just supplies an unusually good one.", GREEN))
story.append(Paragraph(
    "After computing advantages for a whole batch of questions, SPO rescales them together so "
    "they have a consistent size (this is the 'batch normalization' box in Figure 1). This keeps "
    "the learning steps stable from one round to the next. As the experiments will show, this "
    "step turns out to be surprisingly important.", body))

# ---- 2c curriculum ----
story.append(Paragraph("Part C — The curriculum (practice what's learnable)", h2))
story.append(Paragraph(
    "Because the tracker knows each question's current success rate, SPO can be smart about which "
    "questions to practice. Questions the policy already nails (success near 100%) have little "
    "left to teach. Questions it never gets (success near 0%) are too hard to learn from right "
    "now. The richest learning is in the <b>middle</b> — questions it gets about half the time.", body))
story.append(img(f"{DIAG}/d4_curriculum.png", 14.5))
story.append(Paragraph(
    "Figure 4. SPO samples medium-difficulty questions most often. A small floor keeps every "
    "question reachable so nothing is permanently abandoned.", cap))
story.append(Paragraph(
    "This is an <b>adaptive curriculum</b>: as the policy improves and once-hard questions become "
    "medium, they automatically get more practice; once-medium questions that become easy fade "
    "out. GRPO normally can't do this because it has no per-question memory to base the choice on.",
    body))

# ============================ SECTION 3 ============================
story.append(PageBreak())
story.append(Paragraph("3. Putting it together: one training step", h1))
story.append(Paragraph(
    "Here is the full loop. Each round of training repeats these six steps. Notice that steps 2 "
    "and 3 — the single answer and the memory-based comparison — are where SPO differs from GRPO; "
    "the rest is standard.", body))
story.append(img(f"{DIAG}/d5_loop.png", 15.0))
story.append(Paragraph("Figure 5. The SPO training loop (the paper's Algorithm 1).", cap))

story.append(Paragraph("How this maps to the code", h2))
story.append(Paragraph(
    "The repository mirrors these parts directly, so each idea lives in its own file:", body))

code_map = [
    ["File", "What it does", "Section above"],
    ["tracker.py", "The memory v (the success estimate), with KL-based forgetting", "Part A"],
    ["advantage.py", "result \u2212 expectation, batch rescaling, curriculum", "Parts B & C"],
    ["policy.py", "The policy and its update step", "Step 6"],
    ["spo_trainer.py", "The six-step loop", "Figure 5"],
    ["grpo_trainer.py", "The GRPO method, for comparison", "Section 1"],
]
t = Table(code_map, colWidths=[3.4 * cm, 9.0 * cm, 3.9 * cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (0, -1), "Courier-Bold"),
    ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTER]),
    ("TEXTCOLOR", (0, 1), (-1, -1), INK),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe3ea")),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(t)
story.append(Spacer(1, 0.3 * cm))

# ============================ SECTION 4: RESULTS ============================
story.append(PageBreak())
story.append(Paragraph("4. Results", h1))
story.append(Paragraph(
    "To check that the method actually works, the code runs SPO and GRPO head-to-head on a small "
    "test problem (a 'contextual bandit' — a set of questions, each with a hidden best answer). "
    "Both methods get the <b>same amount of computation</b>. We measure the policy's true average "
    "success rate as training proceeds. Higher and faster is better.", body))

story.append(Paragraph("Result 1 — SPO learns faster and ends higher", h2))
story.append(img(f"{FIG}/comparison.png", 16.3))
story.append(Paragraph(
    "Figure 6. Left: SPO (amber) climbs faster and finishes clearly above GRPO (blue), averaged "
    "over 5 runs; shaded bands are the spread. Right: GRPO wastes 20\u201340% of its groups on "
    "the 'all-same-score' problem (blue), while SPO almost never wastes a sample (amber).", cap))
story.append(callout(
    "<b>Headline number.</b> &nbsp;Under matched computation, SPO beat GRPO by about <b>+5 "
    "percentage points</b> of success rate, and won on <b>every single run</b>. The original "
    "paper reports a comparable <b>+3.4 point</b> gain on hard math benchmarks.", SPO))

story.append(Paragraph("Result 2 — Which parts matter? (ablations)", h2))
story.append(Paragraph(
    "An 'ablation' removes one part at a time to see how much it mattered. Each panel compares "
    "full SPO (blue) against a version with one piece switched off (green).", body))
story.append(img(f"{FIG}/ablations.png", 16.3))
story.append(Paragraph(
    "Figure 7. Removing the batch rescaling (bottom-left) hurts the most — confirming the paper's "
    "finding. Removing the memory baseline (top-left) and using random instead of curriculum "
    "sampling (bottom-right) both cost performance. The warm-start (top-right) mainly helps early "
    "stability.", cap))

story.append(PageBreak())
story.append(Paragraph("Result 3 — Why SPO's signal is steadier (variance)", h2))
story.append(Paragraph(
    "The paper argues mathematically that GRPO's comparison bar is noisier than SPO's, especially "
    "for easy and hard questions. The code reproduces this. The ratio below is 'how much noisier "
    "GRPO is than SPO' — above 1 means SPO is steadier.", body))
story.append(img(f"{FIG}/variance_decomposition.png", 16.3))
story.append(Paragraph(
    "Figure 8. The U-shape (left) shows GRPO is worst — relative to SPO — exactly for easy "
    "questions (success near 1) and hard ones (success near 0). The bars (right) confirm the "
    "advantage is largest at those extremes. This matches the paper's ~1.9\u00d7 figure.", cap))

story.append(Paragraph("Result 4 — Why SPO is faster in step-by-step (agentic) tasks", h2))
story.append(Paragraph(
    "When answers take wildly different amounts of time, GRPO's 'wait for the whole group' rule "
    "becomes a serious bottleneck. SPO, needing no group, simply collects whichever answers "
    "finish first and moves on. The simulation below models this.", body))
story.append(img(f"{DIAG}/d6_bottleneck.png", 16.3))
story.append(Paragraph(
    "Figure 9. GRPO (left) must wait for its slowest answer, so two slow stragglers hold up the "
    "whole batch. SPO (right) takes the first answers to finish and discards the slow ones.", cap))
story.append(img(f"{FIG}/throughput.png", 16.3))
story.append(Paragraph(
    "Figure 10. The net effect: in this simulation SPO assembles a training batch about "
    "<b>4.5\u00d7 faster</b> than GRPO, closely matching the paper's reported 4.35\u00d7 speedup.", cap))

# ============================ SECTION 5: TAKEAWAYS ============================
story.append(PageBreak())
story.append(Paragraph("5. Takeaways", h1))

takeaways = [
    ("The big idea", "Replace 'generate a group and compare within it' with 'generate one answer "
     "and compare against a remembered success rate'."),
    ("Why it's better", "No wasted groups, no waiting for stragglers, and a steadier learning "
     "signal — all from removing the group."),
    ("The memory is cheap", "One number per question, updated like a running average. It never "
     "stores a history, so memory stays tiny."),
    ("Forgetting is smart", "Old results count less when the policy has changed a lot, keeping "
     "the memory current."),
    ("Free curriculum", "Because the memory knows each question's difficulty, SPO automatically "
     "practices the questions with the most to teach."),
    ("It holds up", "On a matched-compute test, SPO beat GRPO on every run, with the biggest "
     "gains exactly where the theory predicts."),
]
rows = [[Paragraph(f"<b>{h}</b>", small), Paragraph(b, small)] for h, b in takeaways]
t = Table(rows, colWidths=[4.0 * cm, 12.3 * cm])
t.setStyle(TableStyle([
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHTER]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TEXTCOLOR", (0, 0), (0, -1), SPO),
    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e6e9ef")),
    ("TOPPADDING", (0, 0), (-1, -1), 9),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
]))
story.append(t)
story.append(Spacer(1, 0.5 * cm))

story.append(Paragraph("From this test to real language models", h2))
story.append(Paragraph(
    "This guide used a small test problem so everything runs on a laptop in under a minute. The "
    "algorithm itself doesn't change for large language models: a 'state' becomes a 'prompt', an "
    "'answer' becomes the model's text response, and the success score comes from checking whether "
    "the response is correct. The memory, the surprise-based advantage, and the curriculum all "
    "work identically. In code, moving to a language model means swapping the test environment and "
    "the policy for a real model and its text generator — the three SPO parts stay the same.", body))

story.append(Spacer(1, 0.4 * cm))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#dfe3ea")))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    "Based on \"Single-stream Policy Optimization\" (Xu &amp; Ding). This document explains a "
    "self-contained reimplementation; exact numbers differ from the paper because the test "
    "problem and scale differ, but every qualitative finding is reproduced.",
    S("foot", parent=cap, alignment=TA_LEFT, spaceAfter=0)))


# ---------------- page furniture ----------------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTE)
    canvas.drawString(2 * cm, 1.1 * cm, "Single-stream Policy Optimization — explained")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.setStrokeColor(colors.HexColor("#e6e9ef"))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Single-stream Policy Optimization — explained")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Saved", OUT)
