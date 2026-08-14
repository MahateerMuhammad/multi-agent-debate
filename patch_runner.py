import re
with open('app/evaluation/experiment_runner.py', 'r') as f:
    code = f.read()

# 1. Update single_llm
code = re.sub(
    r'reported_conf = float\(getattr\(data, "confidence", 1\.0\)\).*?number_of_debate_rounds=0,',
    '''raw_conf = getattr(data, "confidence", None)
        reported_conf = float(raw_conf) if raw_conf is not None else None
        conf_status = ConfidenceStatus.reported if reported_conf is not None else ConfidenceStatus.missing
        evaluator_conf = round(0.70 + (reasoning_score * 0.20), 4)

        prompt_toks = llm_resp.usage.prompt_tokens
        comp_toks = llm_resp.usage.completion_tokens
        tok = llm_resp.usage.total_tokens
        
        usage_available = prompt_toks is not None
        prompt_cost = estimate_llm_cost(prompt_toks or 0, 0) if usage_available else None
        comp_cost = estimate_llm_cost(0, comp_toks or 0) if usage_available else None
        cost = (prompt_cost + comp_cost) if usage_available else None

        cost_source = CostSource.local_pricing_estimate if usage_available else CostSource.unavailable
        usage_source = (
            UsageSource.simulated
            if self.eval_mode == EvaluationMode.synthetic
            else (UsageSource.provider_reported if usage_available else UsageSource.unavailable)
        )

        metrics = MetricResult(
            correctness=correctness,
            correctness_status=correctness_status,
            retrieval_recall=retrieval_recall,
            reasoning_lexical_alignment=reasoning_score,
            evidence_grounding=round(grounding, 4),
            citation_source_quality=round(citation_quality, 4),
            completeness=completeness,
            reported_confidence=reported_conf,
            confidence_status=conf_status,
            evaluator_confidence_score=evaluator_conf,
            latency_seconds=latency,
            usage_available=usage_available,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=tok,
            usage_source=usage_source,
            cost_source=cost_source,
            estimated_cost_usd=round(cost, 6) if cost is not None else None,
            number_of_llm_calls=1,
            number_of_debate_rounds=0,
            llm_call_traces=[{
                "node": "proponent",
                "round": 0,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": tok,
                "usage_source": usage_source.value,
            }] if usage_available else []''',
    code, flags=re.DOTALL)

# 2. Update run_two_agent_debate usage accumulation
code = re.sub(
    r'total_toks \+= prop_res\.usage\.total_tokens\s+p_toks \+= prop_res\.usage\.prompt_tokens\s+c_toks \+= prop_res\.usage\.completion_tokens',
    '''if prop_res.usage.prompt_tokens is not None:
                p_toks += prop_res.usage.prompt_tokens
                c_toks += prop_res.usage.completion_tokens or 0
                total_toks += prop_res.usage.total_tokens or (prop_res.usage.prompt_tokens + (prop_res.usage.completion_tokens or 0))''',
    code)
code = re.sub(
    r'total_toks \+= opp_res\.usage\.total_tokens\s+p_toks \+= opp_res\.usage\.prompt_tokens\s+c_toks \+= opp_res\.usage\.completion_tokens',
    '''if opp_res.usage.prompt_tokens is not None:
                p_toks += opp_res.usage.prompt_tokens
                c_toks += opp_res.usage.completion_tokens or 0
                total_toks += opp_res.usage.total_tokens or (opp_res.usage.prompt_tokens + (opp_res.usage.completion_tokens or 0))''',
    code)

# Update run_two_agent_debate traces logic to add llm_call_traces
code = re.sub(
    r'round_traces = \[\]\n\n        for r in range\(rounds\):',
    r'''round_traces = []
        llm_call_traces = []\n
        for r in range(rounds):''',
    code)

code = re.sub(
    r'opp_res = await opponent\.construct_rebuttal\(.*?prop_res\.data\n            \).*?c_toks \+= opp_res\.usage\.completion_tokens or 0\)\)',
    r'''opp_res = await opponent.construct_rebuttal(
                round_topic, proponent_argument=prop_res.data
            )
            if opp_res.usage.prompt_tokens is not None:
                p_toks += opp_res.usage.prompt_tokens
                c_toks += opp_res.usage.completion_tokens or 0
                total_toks += opp_res.usage.total_tokens or (opp_res.usage.prompt_tokens + (opp_res.usage.completion_tokens or 0))

            if prop_res.usage.prompt_tokens is not None:
                llm_call_traces.append({
                    "node": "proponent",
                    "round": r + 1,
                    "prompt_tokens": prop_res.usage.prompt_tokens,
                    "completion_tokens": prop_res.usage.completion_tokens,
                    "total_tokens": prop_res.usage.total_tokens
                })
            if opp_res.usage.prompt_tokens is not None:
                llm_call_traces.append({
                    "node": "opponent",
                    "round": r + 1,
                    "prompt_tokens": opp_res.usage.prompt_tokens,
                    "completion_tokens": opp_res.usage.completion_tokens,
                    "total_tokens": opp_res.usage.total_tokens
                })''',
    code, flags=re.DOTALL)


# 2b Update run_two_agent_debate metrics
code = re.sub(
    r'reported_conf = float\(getattr\(last_prop, "confidence", 1\.0\)\).*?number_of_debate_rounds=rounds,',
    '''raw_conf = getattr(last_prop, "confidence", None)
        reported_conf = float(raw_conf) if raw_conf is not None else None
        conf_status = ConfidenceStatus.reported if reported_conf is not None else ConfidenceStatus.missing
        evaluator_conf = round(0.60 + (reasoning_score * 0.25), 4)

        usage_available = p_toks > 0
        p_cost = estimate_llm_cost(p_toks, 0) if usage_available else None
        c_cost = estimate_llm_cost(0, c_toks) if usage_available else None
        cost = (p_cost + c_cost) if usage_available else None

        cost_source = CostSource.local_pricing_estimate if usage_available else CostSource.unavailable
        usage_source = (
            UsageSource.simulated
            if self.eval_mode == EvaluationMode.synthetic
            else (UsageSource.provider_reported if usage_available else UsageSource.unavailable)
        )

        metrics = MetricResult(
            correctness=correctness,
            correctness_status=correctness_status,
            retrieval_recall=retrieval_recall,
            reasoning_lexical_alignment=reasoning_score,
            evidence_grounding=round(grounding, 4),
            citation_source_quality=round(citation_quality, 4),
            completeness=completeness,
            reported_confidence=reported_conf,
            confidence_status=conf_status,
            evaluator_confidence_score=evaluator_conf,
            latency_seconds=latency,
            usage_available=usage_available,
            prompt_tokens=p_toks if usage_available else None,
            completion_tokens=c_toks if usage_available else None,
            total_tokens=total_toks if usage_available else None,
            usage_source=usage_source,
            cost_source=cost_source,
            estimated_cost_usd=round(cost, 6) if cost is not None else None,
            number_of_llm_calls=rounds * 2,
            number_of_debate_rounds=rounds,
            llm_call_traces=llm_call_traces,''',
    code, flags=re.DOTALL)

with open('app/evaluation/experiment_runner.py', 'w') as f:
    f.write(code)
print("patched 1 and 2")
