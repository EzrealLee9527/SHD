## Reviewer e2ma
<!-- 认为babyllama已经在几个任务上超过了T5和OPT，所以效果不够显著，论文中能说明这一点会更好
希望论文中能加入一个明确的算法流程，包括输入、输出、获得蒸馏模型的步骤，期望这个算法流程能回答到：同时结合模型中的所有头吗？还是每层的所有头一次性结合？如果不是，迭代结合是如何进行的？
在表1中，KD+SHD的损失函数是什么？它是否涉及logit蒸馏项和中间注意力图转换蒸馏项？
可以描述一下将MiniLLM与SHD结合的具体含义吗？即损失函数是如何结合的？以及BabyLlama+SHD是什么意思？它是与BabyLlama预训练相同的过程，但使用SHD损失而不是标准的KL散度损失吗？
承诺会补充多个种子的实验
将图2从附录移到主文本中会非常有用

所有层中的头是同时结合的吗？还是迭代结合的？该方法是否总是生成每层一个注意力头？
SHD是如何与KD、BabyLlama和MiniLLM结合的？在这些情况下的损失函数是什么？
能否提供一些关于压缩率的更多细节？例如，在表1中，当你蒸馏到一个有33M参数的学生模型时，这是否总是对教师架构进行修改，每层只有一个注意力头（如果这确实是SHD的工作方式）？
论文中的结果看起来不错，但我仍然对方法的细节以及它如何与其他损失函数结合感到不确定。如果作者能够令人满意地回答这些问题并提供一些清晰度，我愿意重新考虑我的评分。 -->

We sincerely thank the reviewer for the constructive feedback. Below we address each concern with corresponding updates/clarifications.

**Q:** How are heads combined? Is merging simultaneous or iterative?  
**A:** The merging is **progressive and iterative** (Sec 4.2). For example, to compress 9→3 heads:

- **Phase 1 (9→5)**: 
  - Merge heads (1+2), (3+4), (5+6), (7+8) 
  - Keep head 9 → 5 heads remain

- **Phase 2 (5→3)**:
  - Merge new heads (1'+2'), (3'+4')
  - Keep head 5' → Final 3 heads

This is now formalized in **Algorithm 1** (added to Sec 4.2).
Algorithm 1: Squeezing-Heads Distillation (SHD) - Progressive Merge
```python
def SHD_progressive_merge(teacher_heads, N):
    """
    Args:
        teacher_heads: List of teacher attention heads [A₁ᵀ, ..., Aₘᵀ] (e.g., M=9)
        N: Target student head count (e.g., N=3, M > N)
    Returns:
        Compressed heads [A₁ˢ, ..., Aₙˢ]
    """
    H = teacher_heads.copy()  # Current head pool
    
    while len(H) > N:
        k = 2  # Fixed merge ratio per iteration
        H_new = []
        
        # Merge adjacent pairs
        for i in range(len(H) // k):
            A_p = H[k*i]
            A_q = H[k*i + 1]
            alpha = compute_alpha(A_p, A_q, X_p, X_q)  # Eq.10
            H_new.append(alpha*A_p + (1-alpha)*A_q)  # Eq.6
        
        # Handle remainder
        if len(H) % k != 0:
            H_new.append(H[-1])  # Keep last unmerged head
        
        H = H_new
    
    return H  # Now contains N heads
```

**Q:** Does SHD always produce one head per layer?  
**A:** No. The student can have **any head count ≤ teacher's** (Sec 5). Experiments include:
- DeiT 6→3 heads (Table 4)
- GPT2-XL 25→12/16/20 heads (Table 3)
- LLaMA 40→32 heads (Table 3)

**Q:** Could you provide some more details about compression ratios? For instance, in table 1 when you distill to a student model with 33M parameters, is this always a modification of the teacher architecture with one attention head per layer? 
**A:** We clarify that SHD does not require compressing to one head per layer. We keep the original MDTv2 series architectures，MDTv2/2-XL(the 675M-parameter teacher) has 16 heads, MDTv2/2-B(the 130M-parameter teacher) has 12 heads, MDTv2/2-S(the 33M-parameter student) has 6 heads. SHD supports arbitrary head counts through progressive merging, the 33M-parameter student in Table 1 is distilled using 16→6 heads merging for MDTv2/2-XL(the 675M-parameter teacher) and using 12→6 heads merging for MDTv2/2-B(the 130M-parameter teacher).

**Q:** How is SHD combined with KD/MiniLLM/BabyLlama?  
<!-- 在表1中，KD+SHD的损失函数是什么？它是否涉及logit蒸馏项和中间注意力图转换蒸馏项？
可以描述一下将MiniLLM与SHD结合的具体含义吗？即损失函数是如何结合的？以及BabyLlama+SHD是什么意思？它是与BabyLlama预训练相同的过程，但使用SHD损失而不是标准的KL散度损失吗？ 
SHD是如何与KD、BabyLlama和MiniLLM结合的？在这些情况下的损失函数是什么？
-->
**A:** 




**Q:** Clarity of writing
**A:** We appreciate your detail and constructive criticism and are committed to improving the manuscript. Thank you for your thorough review and suggestions. We have moved Figure 2 from the appendix to the main text to help readers better understand our method.


**Q:** Empirical Robustness
**A:** We will add 95% CIs (3 seeds) to Tables 2/4/6.

We appreciate the reviewers' time and welcome further discussion.

---

## Reviewer KWxu
需要新增更多方法的对比讨论和实验

## Reviewer 2AtZ
<!-- 异构架构的适用性：当教师和学生模型具有不同架构（例如CNN教师和变压器学生）时，SHD的表现如何？
与OFAKD整合：是否考虑过整合OFAKD中的特征投影技术以增强SHD在跨架构蒸馏场景中的表现？
可扩展性：SHD在极小的学生模型（例如10倍小的学生模型）中的表现如何？ -->
Thank you for your insightful comments regarding the applicability of our method to heterogeneous architectures. We would like to clarify our approach and the scope of our work.
**Q:** Applicability to Heterogeneous Architectures: How does SHD perform when the teacher and student models have different architectures, such as a CNN teacher and a Transformer student?​ Have you considered integrating feature projection techniques from OFAKD to enhance SHD's performance in cross-architecture distillation scenarios?​
**A:** Our primary focus is on distillation between transformer models. Heterogeneous architectures, such as CNNs, do not inherently contain attention maps. While it is theoretically possible to create multiple attention maps from intermediate layer features for alignment purposes, this would diverge from our goal of providing an almost cost-free method that does not add parameters or increase training time. Given that transformers have become the mainstream models in many applications, we believe that our method already addresses a significant portion of the current landscape by focusing on homogeneous transformer architectures.
Additionally, we chose not to integrate feature projection techniques from OFAKD for the same reason. Our aim was to maintain an almost cost-free method, and incorporating these techniques would have compromised that objective.

**Q:** Scalability: How does SHD perform in extremely small student models (e.g., a 10x smaller student)?​
**A:** We have experiments using 10x smaller student models in the table 1 (675M teacher and 33M student) and table 3 (1.5B teacher and 120M student). The results show that SHD can still effectively compress attention heads and maintain good performance in such cases.

We appreciate the reviewers' time and welcome further discussion.

## Reviewer UH7y
Thank you for your insightful comments and for highlighting areas in our manuscript that required further clarification. 
**Q:** Can SHD compress any number of heads?  
**A:** Yes. The closed-form solution (Eq.9) allows **arbitrary compression ratios** via iterative merging. For example, to compress 9→3 heads:
- **Phase 1 (9→5)**: 
  - Merge heads (1+2), (3+4), (5+6), (7+8) 
  - Keep head 9 → 5 heads remain

- **Phase 2 (5→3)**:
  - Merge new heads (1'+2'), (3'+4')
  - Keep head 5' → Final 3 heads

This is now formalized in **Algorithm 1** (added to Sec 4.2).
Algorithm 1: Squeezing-Heads Distillation (SHD) - Progressive Merge
```python
def SHD_progressive_merge(teacher_heads, N):
    """
    Args:
        teacher_heads: List of teacher attention heads [A₁ᵀ, ..., Aₘᵀ] (e.g., M=9)
        N: Target student head count (e.g., N=3, M > N)
    Returns:
        Compressed heads [A₁ˢ, ..., Aₙˢ]
    """
    H = teacher_heads.copy()  # Current head pool
    
    while len(H) > N:
        k = 2  # Fixed merge ratio per iteration
        H_new = []
        
        # Merge adjacent pairs
        for i in range(len(H) // k):
            A_p = H[k*i]
            A_q = H[k*i + 1]
            alpha = compute_alpha(A_p, A_q, X_p, X_q)  # Eq.10
            H_new.append(alpha*A_p + (1-alpha)*A_q)  # Eq.6
        
        # Handle remainder
        if len(H) % k != 0:
            H_new.append(H[-1])  # Keep last unmerged head
        
        H = H_new
    
    return H  # Now contains N heads
```

The student can have **any head count ≤ teacher's** (Sec 5). Experiments include:
- DeiT 6→3 heads (Table 4)
- GPT2-XL 25→12/16/20 heads (Table 3)
- LLaMA 40→32 heads (Table 3)

**Q:** Regarding missing definitions.
**A:** We have added definitions in Equation 12 and corrected the wording error in line 145 in the revised version.

We appreciate the reviewers' time and welcome further discussion.

---

