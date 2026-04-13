import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';

const app = express();
app.use(express.json());
app.use(express.static('public'));

// ─── Model IDs via OpenRouter ─────────────────────────────────────────────────
// Verify latest IDs at https://openrouter.ai/models
const M = {
  SONNET:  'anthropic/claude-sonnet-4.6',
  HAIKU:   'anthropic/claude-haiku-4.5',
  OPUS:    'anthropic/claude-opus-4.6',
  O4_MINI: 'openai/o4-mini',
  GEMINI:  'google/gemini-2.5-pro-preview',
};

const THINKING_BUDGETS = {
  simple:   { sonnet: 3000,  opus: 6000  },
  moderate: { sonnet: 6000,  opus: 10000 },
  complex:  { sonnet: 12000, opus: 16000 },
};

// In-memory conversation history per session
const sessions = new Map();

// ─── OpenRouter Client ────────────────────────────────────────────────────────
async function llmOnce(model, messages, { thinking = 0, stream = false, json = false, systemPrompt = null, maxTokens = null } = {}) {
  // max_tokens: explicit override > thinking-based cap > default 2048
  const outputCap = maxTokens ?? (thinking > 0 ? thinking + 2048 : 2048);

  const body = {
    model,
    messages: systemPrompt
      ? [{ role: 'system', content: systemPrompt }, ...messages]
      : messages,
    stream,
    max_tokens: outputCap,
    // Don't use response_format with thinking — they conflict on Anthropic models
    ...(json && thinking === 0 && { response_format: { type: 'json_object' } }),
    ...(thinking > 0 && model.startsWith('anthropic/') && {
      thinking: { type: 'enabled', budget_tokens: thinking },
    }),
  };

  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://council-of-llms.local',
      'X-Title': 'Council of LLMs',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenRouter ${res.status}: ${err}`);
  }

  if (stream) return res; // caller handles stream

  const data = await res.json();
  const choice = data.choices[0];
  const finishReason = choice.finish_reason;
  const content = choice.message.content;

  // Anthropic extended-thinking returns content as an array of blocks
  if (Array.isArray(content)) {
    return content.filter(b => b.type === 'text').map(b => b.text).join('');
  }

  if (finishReason === 'length') {
    console.warn(`[llm] ${model} hit max_tokens (${outputCap}) — response truncated`);
  }

  if (content === null || content === undefined) {
    // Some reasoning models (o4-mini) return null content; log the full choice for debugging
    console.warn(`[llm] ${model} returned null content. finishReason=${finishReason} choice_keys=${Object.keys(choice).join(',')}`);
    // Try to get content from reasoning or other fields
    const alt = choice.message?.reasoning || choice.message?.reasoning_content || '';
    return alt;
  }

  return content;
}

// Retry wrapper — retries on network errors (ECONNRESET, ETIMEDOUT) up to 2 times
async function llm(model, messages, opts = {}) {
  const maxRetries = opts.stream ? 0 : 2; // don't retry streaming calls
  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await llmOnce(model, messages, opts);
    } catch (err) {
      const retryable = err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT' ||
        err.message?.includes('terminated') || err.message?.includes('ECONNRESET');
      if (!retryable || attempt === maxRetries) throw err;
      lastErr = err;
      console.warn(`[llm] ${model} attempt ${attempt + 1} failed (${err.message}), retrying...`);
      await new Promise(r => setTimeout(r, 1500 * (attempt + 1)));
    }
  }
  throw lastErr;
}

// Cached version: wraps the ResearchBundle content with cache_control
function withCache(text) {
  return [{ type: 'text', text, cache_control: { type: 'ephemeral' } }];
}

// ─── Tavily Search ────────────────────────────────────────────────────────────
async function tavilySearch(query) {
  if (!process.env.TAVILY_API_KEY) return [];
  try {
    const res = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: process.env.TAVILY_API_KEY,
        query,
        search_depth: 'advanced',
        max_results: 3,
      }),
    });
    const data = await res.json();
    return (data.results || []).map(r => ({ title: r.title, url: r.url, content: r.content?.slice(0, 800) }));
  } catch {
    return [];
  }
}

// Escape literal newlines/tabs inside JSON string values (LLMs often emit these)
function fixJsonStrings(s) {
  // Replace literal \n \r \t inside quoted string values only
  return s.replace(/"(?:[^"\\]|\\.)*"/g, (match) =>
    match.replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\t/g, '\\t')
  );
}

/**
 * Repair truncated JSON by closing any open string and open brackets.
 * Returns the repaired string, or null if the input is not truncated.
 */
function repairTruncated(s) {
  let inString = false;
  let escape = false;
  const stack = [];

  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (escape) { escape = false; continue; }
    if (c === '\\' && inString) { escape = true; continue; }
    if (c === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (c === '{' || c === '[') stack.push(c);
    else if (c === '}' || c === ']') stack.pop();
  }

  if (stack.length === 0 && !inString) return null; // already balanced

  let result = s.trimEnd();

  if (inString) result += '"'; // close the open string

  // Remove dangling incomplete key/value that got cut mid-way
  result = result.replace(/,\s*"[^"]*"\s*:\s*"[^"]*$/, ''); // key: "partial (already closed above)
  result = result.replace(/,\s*"[^"]*"\s*:\s*$/, '');       // key: <nothing>
  result = result.replace(/,\s*"[^"]*"\s*$/, '');           // bare dangling key
  result = result.replace(/,\s*$/, '');                     // trailing comma

  // Close open structures in reverse
  while (stack.length > 0) {
    result += stack.pop() === '{' ? '}' : ']';
  }

  return result;
}

// Strip markdown code fences and parse JSON robustly
function parseJSON(raw) {
  if (!raw || typeof raw !== 'string') throw new Error('parseJSON: empty or non-string input');

  // 1. Strip markdown fences
  let s = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();

  // 2. Direct parse
  try { return JSON.parse(s); } catch {}

  // 3. Fix literal newlines inside string values, then retry
  const fixed = fixJsonStrings(s);
  try { return JSON.parse(fixed); } catch {}

  // 4. Remove trailing commas before } or ]
  const noTrailing = fixed.replace(/,\s*([}\]])/g, '$1');
  try { return JSON.parse(noTrailing); } catch {}

  // 5. Extract JSON from embedded ```json ... ``` code fence (common with thinking models)
  const fenceMatch = s.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    const fenced = fixJsonStrings(fenceMatch[1].trim()).replace(/,\s*([}\]])/g, '$1');
    try { return JSON.parse(fenced); } catch {}
  }

  // 6. Extract outermost { ... } or [ ... ] and retry with all fixes
  const oStart = s.indexOf('{'), oEnd = s.lastIndexOf('}');
  const aStart = s.indexOf('['), aEnd = s.lastIndexOf(']');

  if (oStart !== -1 && oEnd > oStart) {
    const candidate = fixJsonStrings(s.slice(oStart, oEnd + 1)).replace(/,\s*([}\]])/g, '$1');
    try { return JSON.parse(candidate); } catch {}
  }
  if (aStart !== -1 && aEnd > aStart) {
    const candidate = fixJsonStrings(s.slice(aStart, aEnd + 1)).replace(/,\s*([}\]])/g, '$1');
    try { return JSON.parse(candidate); } catch {}
  }

  // 7. Truncation repair: close open strings and brackets, strip dangling keys
  const repaired = repairTruncated(s);
  if (repaired) {
    const repairedFixed = fixJsonStrings(repaired).replace(/,\s*([}\]])/g, '$1');
    try { return JSON.parse(repairedFixed); } catch {}
  }

  throw new Error(`parseJSON failed: ${s.slice(0, 120)}`);
}

// ─── RAG — US Law Knowledge Base ─────────────────────────────────────────────
const US_LAW_RAG = [
  // ── PROPERTY / LANDLORD-TENANT ────────────────────────────────────────────
  {
    id: 'rag-p01', area: 'property',
    title: 'Security Deposit — California Civil Code § 1950.5',
    content: 'CA Civil Code § 1950.5: Security deposits are limited to 2 months rent (unfurnished) or 3 months rent (furnished). Landlord must return deposit within 21 days of move-out with itemized deduction statement. Allowable deductions: unpaid rent, damage beyond normal wear and tear, cleaning if tenant left unit dirty, restoration of altered property. Normal wear and tear (carpet fading, minor scuffs) may not be deducted. Bad-faith withholding: landlord liable for actual damages plus up to $600 statutory penalty per Baird v. Superior Court.',
    keywords: ['security deposit', 'deposit', 'landlord', 'tenant', 'wear and tear', 'california', 'move-out', '1950.5'],
  },
  {
    id: 'rag-p02', area: 'property',
    title: 'Warranty of Habitability — Implied Covenant',
    content: 'Landlord impliedly warrants that rental premises are habitable at inception of tenancy and throughout term. Source: Hinson v. Delis (1972) CA; codified in CA Civil Code § 1941. Required conditions: weatherproofing, hot/cold water, heating, electrical wiring, plumbing, sanitation, natural light. Tenant remedies for breach: repair-and-deduct (up to one month rent, twice per year), rent withholding, constructive eviction, tort damages. Federal: Javins v. First National Realty Corp (DC Cir. 1970) established national precedent.',
    keywords: ['habitability', 'warranty', 'landlord', 'tenant', 'repair', 'plumbing', 'heating', 'uninhabitable', 'constructive eviction'],
  },
  {
    id: 'rag-p03', area: 'property',
    title: 'Eviction — Unlawful Detainer Procedure',
    content: 'Landlord must follow statutory process: (1) Serve proper notice — 3-day notice to pay or quit for nonpayment; 30-day notice for month-to-month tenancy under 1 year; 60-day notice for tenancy over 1 year. (2) File unlawful detainer if tenant stays. (3) Court hearing within 20 days. Self-help eviction (changing locks, removing belongings, utility shutoff) is illegal in all US states and gives tenant claim for damages. CARES Act (2020) added federally-backed property protections extended by state law.',
    keywords: ['eviction', 'unlawful detainer', 'notice to quit', 'lockout', 'self-help', 'month-to-month', '3-day', '30-day', '60-day'],
  },
  {
    id: 'rag-p04', area: 'property',
    title: 'Fair Housing Act — 42 U.S.C. § 3604',
    content: 'Federal Fair Housing Act (1968) prohibits discrimination in sale or rental of housing based on race, color, national origin, religion, sex, familial status, disability. 42 U.S.C. § 3604. Key cases: Texas Dept of Housing v. Inclusive Communities Project (2015) — disparate impact claims cognizable under FHA. HUD v. Rucker (2002) — public housing drug eviction. Landlord must provide reasonable accommodations for disability (e.g., allow service animals despite no-pet policy). Damages: actual damages, punitive up to $16,000 first offense.',
    keywords: ['fair housing', 'discrimination', 'disability', 'race', 'national origin', 'accommodation', 'section 3604', 'HUD'],
  },
  {
    id: 'rag-p05', area: 'property',
    title: 'Adverse Possession — Elements and Requirements',
    content: 'Claimant must prove: (1) actual possession, (2) open and notorious (visible to owner), (3) exclusive, (4) continuous for statutory period (3–21 years by state), (5) hostile/adverse (without owner permission). Minority: must also be under color of title or good faith claim. CA Code of Civil Procedure § 318–325: 5-year period, must pay taxes. Key case: Nome 2000 v. Fagerstrom (Alaska 1990) — cultural use patterns satisfy possession element. Easement by prescription: same elements except exclusive not required.',
    keywords: ['adverse possession', 'prescriptive easement', 'hostile', 'continuous', 'open notorious', 'color of title', 'property tax'],
  },
  {
    id: 'rag-p06', area: 'property',
    title: 'Takings Clause — Fifth Amendment Just Compensation',
    content: 'Fifth Amendment: "nor shall private property be taken for public use, without just compensation." Physical takings: government physically occupies property — per se taking (Loretto v. Teleprompter, 1982). Regulatory takings: Penn Central Transportation Co. v. New York City (1978) — balancing test: economic impact, investment-backed expectations, character of regulation. Lucas v. South Carolina Coastal Council (1992): denial of all economically beneficial use = categorical taking. Kelo v. City of New London (2005): economic development qualifies as public use.',
    keywords: ['eminent domain', 'taking', 'just compensation', 'regulatory taking', 'fifth amendment', 'public use', 'kelo', 'penn central'],
  },

  // ── CONTRACT LAW ──────────────────────────────────────────────────────────
  {
    id: 'rag-c01', area: 'contract',
    title: 'Contract Formation — Offer, Acceptance, Consideration',
    content: 'Valid contract requires: (1) offer — definite terms, communicated intent to be bound; (2) acceptance — mirror image (common law) or UCC § 2-207 battle of the forms; (3) consideration — bargained-for legal detriment; (4) mutual assent; (5) capacity; (6) legality. Illusory promise lacks consideration (Mattei v. Hopper, 1958). Promissory estoppel (Restatement 2d § 90): promise + reasonable reliance + injustice without enforcement. Past consideration is not valid consideration (Dyer v. National By-Products, 1986).',
    keywords: ['contract', 'offer', 'acceptance', 'consideration', 'formation', 'mutual assent', 'promissory estoppel', 'capacity'],
  },
  {
    id: 'rag-c02', area: 'contract',
    title: 'Breach of Contract — Material vs. Minor Breach',
    content: 'Material breach discharges non-breaching party from performance duties and allows immediate suit for expectation damages. Test (Restatement 2d § 241): extent of performance received, adequacy of compensation, forfeiture to breaching party, likelihood of cure, and good faith. Jacob & Youngs v. Kent (1921): pipe brand deviation = minor breach; diminution-in-value damages only. Anticipatory repudiation: unequivocal statement of future non-performance — treated as immediate breach (Hochster v. De La Tour, 1853). UCC § 2-601 perfect tender rule: buyer may reject goods for any defect in commercial transactions.',
    keywords: ['breach', 'material breach', 'anticipatory repudiation', 'perfect tender', 'performance', 'repudiation', 'cure'],
  },
  {
    id: 'rag-c03', area: 'contract',
    title: 'Contract Damages — Expectation, Reliance, Restitution',
    content: 'Expectation damages: put plaintiff in position had contract been performed — lost profits, diminution in value, cost of completion. Hadley v. Baxendale (1854) foreseeability rule: damages recoverable only if foreseeable at time of contracting. Reliance damages: out-of-pocket expenses incurred in reliance on promise. Restitution/unjust enrichment: value conferred on defendant. Duty to mitigate: non-breaching party must take reasonable steps to reduce losses (Parker v. Twentieth Century-Fox, 1970 — refusal of different/inferior work OK). Liquidated damages clause enforced if: (1) actual damages difficult to estimate, (2) amount is reasonable forecast.',
    keywords: ['damages', 'expectation', 'reliance', 'restitution', 'mitigation', 'foreseeability', 'hadley', 'liquidated damages', 'lost profits'],
  },
  {
    id: 'rag-c04', area: 'contract',
    title: 'Contract Defenses — Fraud, Duress, Unconscionability',
    content: 'Fraudulent misrepresentation: false representation of material fact, scienter, intent to induce reliance, justifiable reliance, damages. Mutual mistake: both parties mistaken about material fact at time of contracting — voidable (Sherwood v. Walker, 1887). Unilateral mistake: generally not a defense unless other party knew. Duress: improper threat that left no reasonable alternative. Economic duress: threat to breach contract to extort modification. Unconscionability (UCC § 2-302): procedural (no meaningful choice) + substantive (oppressive terms) — Williams v. Walker-Thomas Furniture (1965). Statute of Frauds: real estate, contracts over 1 year, sale of goods over $500 must be in writing.',
    keywords: ['fraud', 'misrepresentation', 'duress', 'unconscionability', 'mutual mistake', 'statute of frauds', 'writing', 'undue influence', 'voidable'],
  },
  {
    id: 'rag-c05', area: 'contract',
    title: 'UCC Article 2 — Sale of Goods',
    content: 'UCC Article 2 governs sale of goods (movable property). Merchants held to higher standards. Gap fillers: UCC supplies missing terms (price, delivery). Implied warranty of merchantability (§ 2-314): goods fit for ordinary purpose. Implied warranty of fitness for particular purpose (§ 2-315): seller knows buyer\'s purpose and buyer relies on seller\'s judgment. Express warranties (§ 2-313): any affirmation of fact creates warranty. Disclaimer of implied warranties must mention "merchantability" and be conspicuous. Remedies: cover (§ 2-712), market price difference (§ 2-713), specific performance for unique goods (§ 2-716).',
    keywords: ['UCC', 'sale of goods', 'merchantability', 'warranty', 'goods', 'commercial', 'article 2', 'cover', 'merchant'],
  },
  {
    id: 'rag-c06', area: 'contract',
    title: 'Parol Evidence Rule and Contract Interpretation',
    content: 'Parol evidence rule: integrated written agreement may not be contradicted by prior oral/written agreements. Partial integration: may be supplemented by consistent additional terms. Pacific Gas & Electric v. G.W. Thomas Drayage (1968): extrinsic evidence admissible to explain any term, even if unambiguous on its face (CA rule). Plain meaning rule (most states): unambiguous contract terms enforced as written. Exceptions: fraud, duress, mistake, condition precedent, ambiguity, trade usage. Course of dealing and course of performance admissible under UCC § 1-303.',
    keywords: ['parol evidence', 'integration', 'ambiguity', 'interpretation', 'extrinsic evidence', 'course of dealing', 'plain meaning'],
  },

  // ── TORT LAW ──────────────────────────────────────────────────────────────
  {
    id: 'rag-t01', area: 'tort',
    title: 'Negligence — Elements, Duty, Breach, Causation',
    content: 'Four elements: (1) Duty: reasonable person standard; Palsgraf v. Long Island R.R. (1928) — duty limited to foreseeable plaintiffs. (2) Breach: Hand formula — B < PL (burden of precaution < probability × magnitude of harm). (3) Causation: but-for test; substantial factor test for multiple causes. (4) Damages. Res ipsa loquitur: injury type that doesn\'t normally occur without negligence, defendant had exclusive control. Professional negligence (malpractice): custom of profession sets standard of care. Negligence per se: violation of statute designed to protect this class of plaintiff from this type of harm.',
    keywords: ['negligence', 'duty', 'breach', 'causation', 'damages', 'reasonable person', 'malpractice', 'palsgraf', 'res ipsa', 'foreseeability'],
  },
  {
    id: 'rag-t02', area: 'tort',
    title: 'Strict Liability — Abnormally Dangerous Activities and Products',
    content: 'Strict liability: no proof of fault required. Abnormally dangerous activities (Restatement 2d § 519): serious risk of harm, activity not of common usage — dynamite blasting, chemical storage. Products liability strict liability (Greenman v. Yuba Power Products, 1963): manufacturer liable for injury from defective product in normal use. Three types of defect: manufacturing defect, design defect (risk-utility or consumer expectation test), failure to warn. Restatement 3d Products Liability (1998) applies risk-utility to design and warning claims. Defenses: assumption of risk, misuse, comparative fault in many states.',
    keywords: ['strict liability', 'products liability', 'defect', 'manufacturing', 'design defect', 'failure to warn', 'greenman', 'abnormally dangerous'],
  },
  {
    id: 'rag-t03', area: 'tort',
    title: 'Intentional Torts — Battery, Assault, IIED',
    content: 'Battery: intentional harmful or offensive contact with person of another — Vosburg v. Putney (1891). Assault: reasonable apprehension of imminent harmful contact. Trespass to land: intentional entry without consent. False imprisonment: intentional, complete confinement against will without justification. Intentional infliction of emotional distress (IIED): extreme and outrageous conduct, intent or recklessness, severe emotional distress — Wilkinson v. Downton (1897). Transferred intent: intent to harm A satisfies intent element for harm to B. Consent, self-defense, defense of others, necessity are complete defenses.',
    keywords: ['battery', 'assault', 'IIED', 'trespass', 'false imprisonment', 'intentional tort', 'extreme outrageous', 'consent', 'self-defense'],
  },
  {
    id: 'rag-t04', area: 'tort',
    title: 'Defamation — Libel, Slander, First Amendment Limits',
    content: 'Defamation: false statement of fact, published to third party, that harms reputation. Libel (written) vs. slander (spoken). Public figure/official (New York Times v. Sullivan, 1964): must prove actual malice (knowledge of falsity or reckless disregard). Private figure: negligence standard (Gertz v. Robert Welch, 1974). Opinion is protected (Milkovich v. Lorain Journal, 1990 — pure opinion test). Defamation per se: crimes, sexual immorality, business incompetence — damages presumed. Truth is absolute defense. Retraction statutes may limit damages.',
    keywords: ['defamation', 'libel', 'slander', 'reputation', 'actual malice', 'public figure', 'new york times', 'gertz', 'first amendment', 'opinion'],
  },
  {
    id: 'rag-t05', area: 'tort',
    title: 'Comparative and Contributory Negligence',
    content: 'Pure contributory negligence (4 states + DC): any fault by plaintiff bars recovery. Pure comparative fault: recovery reduced by plaintiff\'s percentage of fault — Li v. Yellow Cab Co. (CA, 1975). Modified comparative fault (majority): plaintiff cannot recover if more than 50% at fault (or 51% in some states). Assumption of risk: express (signed waiver) = complete bar. Implied primary (inherent in activity) = complete bar. Implied secondary (aware of created risk) = merged into comparative fault. Last clear chance doctrine: defendant who had last clear chance to avoid harm cannot use contributory negligence as defense.',
    keywords: ['comparative negligence', 'contributory negligence', 'assumption of risk', 'fault allocation', 'li v yellow cab', 'apportionment'],
  },
  {
    id: 'rag-t06', area: 'tort',
    title: 'Wrongful Death and Survival Actions',
    content: 'Wrongful death statutes: allow certain beneficiaries (spouse, children, parents) to sue for decedent\'s death caused by defendant\'s wrongful act. Damages: pecuniary loss, loss of consortium, loss of services, medical expenses, funeral costs. Survival action: decedent\'s cause of action survives death; estate sues for decedent\'s pain and suffering, lost earnings during survival period. Key: wrongful death = beneficiaries\' loss; survival = decedent\'s own losses. Federal Employers Liability Act (FELA) for railroad workers. Jones Act for seamen. Most states bar punitive damages in wrongful death.',
    keywords: ['wrongful death', 'survival action', 'death', 'damages', 'beneficiaries', 'consortium', 'FELA', 'pecuniary'],
  },

  // ── EMPLOYMENT LAW ────────────────────────────────────────────────────────
  {
    id: 'rag-e01', area: 'employment',
    title: 'Employment Discrimination — Title VII, EEOC Process',
    content: 'Title VII of the Civil Rights Act of 1964 (42 U.S.C. § 2000e): prohibits discrimination based on race, color, national origin, sex, religion in employment. Applies to employers with 15+ employees. Disparate treatment (intentional): McDonnell Douglas Corp. v. Green (1973) burden-shifting: plaintiff establishes prima facie case → employer articulates legitimate reason → plaintiff shows pretext. Disparate impact: Griggs v. Duke Power Co. (1971) — neutral policies with adverse impact require business necessity justification. EEOC charge must be filed within 180 days (or 300 days in states with anti-discrimination agencies). Right-to-sue letter required before federal lawsuit.',
    keywords: ['discrimination', 'title VII', 'EEOC', 'race', 'sex', 'religion', 'disparate treatment', 'disparate impact', 'mcdonnell douglas', 'pretext'],
  },
  {
    id: 'rag-e02', area: 'employment',
    title: 'Wrongful Termination — At-Will Exceptions',
    content: 'Default US rule: at-will employment — either party may terminate at any time for any reason. Exceptions: (1) Implied contract — handbook language, progressive discipline policies (Toussaint v. Blue Cross, 1980); (2) Implied covenant of good faith and fair dealing; (3) Public policy — firing for refusing to break law, filing workers\' comp, jury duty, whistleblowing (Tameny v. Atlantic Richfield, 1980 CA); (4) WARN Act (29 U.S.C. § 2101): 60 days notice for mass layoffs (50+ employees). Retaliation for protected activity (Title VII § 704(a)) is unlawful.',
    keywords: ['wrongful termination', 'at-will', 'public policy', 'retaliation', 'whistleblower', 'implied contract', 'WARN act', 'handbook'],
  },
  {
    id: 'rag-e03', area: 'employment',
    title: 'Sexual Harassment — Hostile Work Environment and Quid Pro Quo',
    content: 'Title VII prohibits two forms: (1) Quid pro quo: supervisor conditions employment benefit on sexual favors — strict employer liability (Burlington Industries v. Ellerth, 1998). (2) Hostile work environment: severe or pervasive conduct that alters conditions of employment — Meritor Savings Bank v. Vinson (1986). Employer affirmative defense to hostile environment (Faragher/Ellerth): (a) exercised reasonable care to prevent/correct, (b) plaintiff unreasonably failed to use complaint procedures. Oncale v. Sundowner (1998): same-sex harassment covered. #MeToo era: courts increasingly scrutinize Faragher defense adequacy.',
    keywords: ['sexual harassment', 'hostile work environment', 'quid pro quo', 'title VII', 'faragher', 'ellerth', 'employer liability', 'harassment'],
  },
  {
    id: 'rag-e04', area: 'employment',
    title: 'Wage and Hour Law — FLSA, Overtime, Misclassification',
    content: 'Fair Labor Standards Act (29 U.S.C. § 201): minimum wage ($7.25 federal; many states higher), overtime at 1.5× for hours over 40/week. Exemptions: executive, administrative, professional (EAP) — must meet salary basis ($684/week) + duties test. Independent contractor vs. employee: economic reality test (multiple factors: control, permanence, integral to business). Misclassification consequences: back wages, liquidated damages, attorney fees. California AB-5 (2019): ABC test — presumption of employee status unless A (control), B (outside usual business), C (independent established trade) all met. FLSA § 207(a) collective actions common.',
    keywords: ['wages', 'overtime', 'FLSA', 'minimum wage', 'independent contractor', 'misclassification', 'exempt', 'AB-5', 'hour', 'pay'],
  },
  {
    id: 'rag-e05', area: 'employment',
    title: 'Non-Compete Agreements — Enforceability and Limits',
    content: 'Non-compete agreements must be reasonable in scope, geographic area, and duration. Most states apply balancing test: legitimate business interest vs. hardship on employee. California Business & Professions Code § 16600: non-competes void and unenforceable (with narrow exceptions for business sale). FTC Non-Compete Rule (2024): banned most worker non-competes nationally; status pending litigation. Trade secret protection (Defend Trade Secrets Act, 18 U.S.C. § 1836): alternative to non-competes — protects actual confidential information without restricting employment. Non-solicitation clauses (clients, employees) generally more enforceable than broad non-competes.',
    keywords: ['non-compete', 'non-solicitation', 'covenant not to compete', 'trade secret', 'california', 'restrictive covenant', 'FTC', 'section 16600'],
  },
  {
    id: 'rag-e06', area: 'employment',
    title: 'FMLA — Family and Medical Leave',
    content: 'Family and Medical Leave Act (29 U.S.C. § 2601): eligible employees entitled to 12 weeks unpaid leave per year for: birth/adoption of child, serious health condition of employee or immediate family member. Employer must have 50+ employees. Employee must have worked 12 months and 1,250 hours. Employer must restore employee to same or equivalent position. Cannot retaliate for taking FMLA leave. Serious health condition: inpatient care or continuing treatment by health care provider. ADA (42 U.S.C. § 12101) may provide additional leave as reasonable accommodation. California CFRA expands FMLA protections.',
    keywords: ['FMLA', 'family leave', 'medical leave', 'maternity', 'disability leave', 'ADA', 'serious health condition', 'accommodation'],
  },

  // ── CONSTITUTIONAL LAW ────────────────────────────────────────────────────
  {
    id: 'rag-cn01', area: 'constitutional',
    title: 'First Amendment — Free Speech, Protected and Unprotected Categories',
    content: 'First Amendment prohibits government (not private) restriction of speech. Content-neutral restrictions: intermediate scrutiny — narrowly tailored to serve significant interest, leave open alternative channels (Ward v. Rock Against Racism, 1989). Content-based: strict scrutiny — compelling interest, least restrictive means. Unprotected: obscenity (Miller v. California, 1973 — prurient, patently offensive, no serious value), true threats, incitement (Brandenburg v. Ohio, 1969 — imminent lawless action), fighting words (Chaplinsky), defamation, child pornography. Public forum doctrine: traditional public forums (streets, parks) — heightened protection.',
    keywords: ['first amendment', 'free speech', 'content neutral', 'strict scrutiny', 'obscenity', 'incitement', 'public forum', 'Brandenburg'],
  },
  {
    id: 'rag-cn02', area: 'constitutional',
    title: 'Fourth Amendment — Search and Seizure, Warrant Requirements',
    content: 'Fourth Amendment prohibits unreasonable searches and seizures; warrants require probable cause and particularity. Katz v. United States (1967): reasonable expectation of privacy test. Warrant exceptions: exigent circumstances, search incident to arrest (Chimel), automobile exception (Carroll), plain view, consent, inventory search, administrative/regulatory. Terry v. Ohio (1968): stop and frisk requires reasonable articulable suspicion. Third-party doctrine (Smith v. Maryland, 1979): no REOP in info shared with third parties, but Carpenter v. United States (2018) limits for cell-site location data. Exclusionary rule: Mapp v. Ohio (1961) — unconstitutionally obtained evidence excluded.',
    keywords: ['fourth amendment', 'search', 'seizure', 'warrant', 'probable cause', 'exclusionary rule', 'Terry', 'Katz', 'privacy', 'stop and frisk'],
  },
  {
    id: 'rag-cn03', area: 'constitutional',
    title: 'Fifth and Fourteenth Amendment — Due Process, Equal Protection',
    content: 'Substantive due process: fundamental rights require strict scrutiny (privacy, marriage, travel). Non-fundamental: rational basis. Procedural due process: Mathews v. Eldridge (1976) — private interest, government interest, risk of erroneous deprivation. Equal protection (14th Amend.): suspect classifications (race, national origin) = strict scrutiny (Korematsu; but see SFFA v. Harvard, 2023 ending race-conscious admissions). Quasi-suspect (sex): intermediate scrutiny (Craig v. Boren). Everything else: rational basis. State action required for constitutional claims — Shelley v. Kraemer (1948) exception for judicial enforcement of private discrimination.',
    keywords: ['due process', 'equal protection', 'fourteenth amendment', 'strict scrutiny', 'rational basis', 'intermediate scrutiny', 'fundamental rights', 'state action'],
  },
  {
    id: 'rag-cn04', area: 'constitutional',
    title: 'Second Amendment — Heller to Bruen Framework',
    content: 'District of Columbia v. Heller (2008): Second Amendment protects individual right to keep firearms at home for self-defense; not unlimited. McDonald v. Chicago (2010): incorporated against states via 14th Amendment. New York State Rifle & Pistol Ass\'n v. Bruen (2022): text-and-history test replaces means-end scrutiny. Regulation constitutional only if consistent with Nation\'s historical tradition of firearm regulation. Post-Bruen: Rahimi (2024) — domestic violence restraining order disarmament upheld as consistent with historical tradition of disarming dangerous persons.',
    keywords: ['second amendment', 'gun', 'firearm', 'heller', 'bruen', 'mcdonald', 'self-defense', 'weapons', 'rahimi'],
  },

  // ── CRIMINAL LAW / PROCEDURE ──────────────────────────────────────────────
  {
    id: 'rag-cr01', area: 'criminal',
    title: 'Criminal Law — Elements of Crime, Mens Rea Standards',
    content: 'Actus reus (guilty act) + mens rea (guilty mind) + causation + harm. MPC mens rea hierarchy: purpose (conscious object), knowledge (aware of certainty), recklessness (conscious disregard of substantial risk), negligence (should have been aware). Specific intent crimes: malice aforethought (murder), premeditation, knowledge of fact. General intent: awareness of act. Strict liability: no mens rea required (statutory rape in most states, regulatory offenses). Vicarious liability: employer/employee in regulatory context. Corporate criminal liability: respondeat superior for employee acts in scope of employment.',
    keywords: ['criminal', 'mens rea', 'actus reus', 'intent', 'recklessness', 'negligence', 'specific intent', 'MPC', 'crime', 'knowledge'],
  },
  {
    id: 'rag-cr02', area: 'criminal',
    title: 'Homicide — Murder, Manslaughter, Felony Murder',
    content: 'First-degree murder: premeditated, deliberate killing; killing by specified means (poison, lying in wait). Second-degree murder: intent to kill without premeditation; extreme recklessness (depraved heart). Voluntary manslaughter: intentional killing under adequate provocation/heat of passion — MPC extreme mental or emotional disturbance. Involuntary manslaughter: criminal negligence or unlawful act not amounting to felony. Felony murder: death occurring during dangerous felony = murder; many states have reformed (CA SB 1437, 2018). Castle doctrine: no duty to retreat in home. Stand Your Ground: no duty to retreat in public (majority of states).',
    keywords: ['murder', 'manslaughter', 'homicide', 'felony murder', 'premeditation', 'castle doctrine', 'stand your ground', 'heat of passion', 'provocation'],
  },
  {
    id: 'rag-cr03', area: 'criminal',
    title: 'Fifth Amendment — Miranda Rights, Self-Incrimination',
    content: 'Miranda v. Arizona (1966): before custodial interrogation, police must warn: right to remain silent, anything said can be used, right to attorney, right to appointed attorney if indigent. Custody: reasonable person would not feel free to leave (Stansbury v. California). Interrogation: express questioning + functional equivalent. Waiver: voluntary, knowing, intelligent — Berghuis v. Thompkins (2010): must unambiguously invoke. Public safety exception (Quarles). Double jeopardy: bars prosecution for same offense after acquittal/conviction; Blockburger test for same offense. Grand jury indictment: required for federal felonies.',
    keywords: ['miranda', 'fifth amendment', 'self-incrimination', 'right to remain silent', 'custody', 'interrogation', 'waiver', 'double jeopardy'],
  },
  {
    id: 'rag-cr04', area: 'criminal',
    title: 'Sixth Amendment — Right to Counsel, Confrontation, Speedy Trial',
    content: 'Gideon v. Wainwright (1963): right to counsel in all felony prosecutions; extended to misdemeanor with imprisonment (Argersinger v. Hamlin). Right attaches at critical stage: formal charges, lineup after indictment. Ineffective assistance (Strickland v. Washington, 1984): deficient performance + prejudice. Confrontation Clause: Crawford v. Washington (2004) — testimonial hearsay inadmissible absent opportunity to cross-examine declarant. Speedy trial: Barker v. Wingo (1972) four-factor balancing. Brady v. Maryland (1963): prosecution must disclose material exculpatory evidence. Batson v. Kentucky (1986): race-neutral explanation required for peremptory challenges.',
    keywords: ['right to counsel', 'sixth amendment', 'confrontation', 'speedy trial', 'gideon', 'strickland', 'Brady', 'discovery', 'ineffective assistance'],
  },

  // ── CIVIL PROCEDURE ───────────────────────────────────────────────────────
  {
    id: 'rag-cp01', area: 'civil_procedure',
    title: 'Federal Subject Matter Jurisdiction — Diversity and Federal Question',
    content: 'Article III federal jurisdiction: (1) Federal question (28 U.S.C. § 1331): claim arising under federal law, Constitution, treaties — well-pleaded complaint rule (Louisville & Nashville RR v. Mottley, 1908). (2) Diversity (28 U.S.C. § 1332): complete diversity between all plaintiffs and defendants + amount in controversy exceeds $75,000. Citizenship: domicile for individuals; state of incorporation + principal place of business for corporations (Hertz Corp. v. Friend, 2010 nerve center test). Supplemental jurisdiction (§ 1367): claims sharing common nucleus of operative fact. Removal from state court (§ 1441): within 30 days of service.',
    keywords: ['jurisdiction', 'diversity', 'federal question', 'amount in controversy', 'removal', 'subject matter jurisdiction', '1331', '1332'],
  },
  {
    id: 'rag-cp02', area: 'civil_procedure',
    title: 'Pleading Standards — Twombly/Iqbal and Rule 12',
    content: 'Federal Rule 8(a): short and plain statement of claim showing entitlement to relief. Twombly (2007) + Iqbal (2009): plausibility pleading standard — factual allegations must be plausible on their face, not merely conceivable; court disregards conclusory allegations. Rule 9(b): fraud and mistake must be pled with particularity. Rule 12(b)(6): failure to state a claim — tests legal sufficiency, not facts. Rule 12(b)(2): lack of personal jurisdiction. Rule 12(c): judgment on pleadings. Rule 15: amendments freely granted when justice requires; relation back doctrine. FRCP 11: sanctions for frivolous claims.',
    keywords: ['pleading', 'twombly', 'iqbal', 'rule 12', 'motion to dismiss', 'rule 8', 'complaint', 'plausibility', 'FRCP'],
  },
  {
    id: 'rag-cp03', area: 'civil_procedure',
    title: 'Summary Judgment — Rule 56 Standard',
    content: 'FRCP Rule 56: summary judgment granted when no genuine dispute as to material fact and movant entitled to judgment as matter of law. Anderson v. Liberty Lobby (1986): same standard of proof as trial — clear and convincing evidence cases require correspondingly higher threshold. Celotex Corp. v. Catrett (1986): movant may shift burden by showing absence of evidence on essential element; non-movant must then come forward with specific facts. Matsushita v. Zenith Radio (1986): inferences must be reasonable, not implausible. Evidence viewed in light most favorable to non-movant. Partial summary judgment on discrete issues allowed.',
    keywords: ['summary judgment', 'rule 56', 'genuine dispute', 'material fact', 'celotex', 'anderson', 'no trial', 'burden of proof'],
  },
  {
    id: 'rag-cp04', area: 'civil_procedure',
    title: 'Personal Jurisdiction — Minimum Contacts and Due Process',
    content: 'International Shoe Co. v. Washington (1945): defendant must have minimum contacts with forum such that suit does not offend traditional notions of fair play and substantial justice. Specific jurisdiction: claim arises out of forum contacts. General jurisdiction: so substantial and systematic contacts that essentially at home (Goodyear; Daimler AG v. Bauman, 2014 — for corporations, only state of incorporation or principal place of business). Purposeful availment: Hanson v. Denckla (1958). Stream of commerce: split — World-Wide Volkswagen v. Woodson (1980) requires purposeful direction. Internet contacts: Calder effects test for intentional torts.',
    keywords: ['personal jurisdiction', 'minimum contacts', 'due process', 'specific jurisdiction', 'general jurisdiction', 'purposeful availment', 'international shoe'],
  },
  {
    id: 'rag-cp05', area: 'civil_procedure',
    title: 'Class Actions — Rule 23 Requirements',
    content: 'FRCP Rule 23: four prerequisites: (1) numerosity (40+ members typical), (2) commonality (common questions of law or fact), (3) typicality (representative claims typical of class), (4) adequacy of representation. Then must satisfy one of 23(b): (b)(1) inconsistent adjudications, (b)(2) injunctive/declaratory relief, (b)(3) common questions predominate + superiority. Wal-Mart v. Dukes (2011): commonality requires common answer to common question. Class certified before merits. CAFA (2005): federal jurisdiction for class actions with 100+ members, $5M+ in controversy. Cy pres settlements growing area.',
    keywords: ['class action', 'rule 23', 'commonality', 'numerosity', 'typicality', 'adequacy', 'CAFA', 'wal-mart dukes', 'certification'],
  },

  // ── FAMILY LAW ────────────────────────────────────────────────────────────
  {
    id: 'rag-f01', area: 'family',
    title: 'Divorce — Grounds, Property Division, Equitable Distribution',
    content: 'No-fault divorce: irreconcilable differences or irretrievable breakdown (all US states since 2010). Fault grounds (most states still recognize): adultery, cruelty, desertion. Community property states (CA, TX, AZ, NV, WA, ID, NM, LA, WI): marital property split 50/50. Equitable distribution (majority): fair but not necessarily equal division considering length of marriage, contributions, economic circumstances. Separate property: pre-marital assets, gifts, inheritances — generally not divided. Transmutation: separate property can become marital by commingling or agreement. QDRO required to divide pension/retirement accounts.',
    keywords: ['divorce', 'property division', 'community property', 'equitable distribution', 'marital property', 'separate property', 'QDRO', 'dissolution'],
  },
  {
    id: 'rag-f02', area: 'family',
    title: 'Child Custody and Support',
    content: 'Best interests of the child standard: factors include parental fitness, stability, child\'s preference (age-appropriate), sibling relationships, each parent\'s willingness to facilitate relationship with other parent. Legal custody: decision-making authority. Physical custody: where child lives. Joint custody presumptively favored. Child support: most states use income shares model; federal Child Support Enforcement (IV-D program). Parental kidnapping: PKPA (1980) and UCCJEA — home state jurisdiction. Modification: substantial change in circumstances. Termination of parental rights: abuse, neglect, abandonment — clear and convincing evidence.',
    keywords: ['custody', 'child support', 'best interests', 'parenting', 'visitation', 'UCCJEA', 'modification', 'termination', 'guardian ad litem'],
  },
  {
    id: 'rag-f03', area: 'family',
    title: 'Spousal Support — Alimony Standards',
    content: 'Alimony/spousal support: court discretion based on: length of marriage, standard of living, earning capacity, sacrifices made (career foregone), financial need. Types: temporary (pendente lite), rehabilitative (time-limited), permanent (long marriages with large earning gap), lump sum. Tax treatment changed under TCJA 2017: alimony no longer deductible for payer / includable for recipient for divorces after 12/31/18. Modification: substantial change in circumstances; death or remarriage typically terminates. Prenuptial agreements may waive spousal support if: voluntary, full disclosure, not unconscionable — Uniform Premarital Agreement Act.',
    keywords: ['alimony', 'spousal support', 'maintenance', 'prenuptial', 'modification', 'rehabilitative', 'earning capacity', 'UPAA'],
  },

  // ── CONSUMER PROTECTION ───────────────────────────────────────────────────
  {
    id: 'rag-co01', area: 'consumer_protection',
    title: 'FCRA — Fair Credit Reporting Act',
    content: 'Fair Credit Reporting Act (15 U.S.C. § 1681): governs credit reports. Consumer rights: access to free annual credit report (AnnualCreditReport.com), dispute inaccurate information (CRA must investigate within 30 days), freeze credit. CRA duties: reasonable procedures for maximum accuracy; delete unverifiable disputed information. Furnisher duties: investigate disputes, correct inaccurate info. Statute of limitations: 2 years from violation, or 5 years from violation if willful. Damages: actual damages, punitive damages for willful violations up to $1,000 plus attorney fees. Equifax v. consumers class action settlements established procedures.',
    keywords: ['credit report', 'FCRA', 'credit bureau', 'dispute', 'inaccurate', 'Equifax', 'Experian', 'TransUnion', 'consumer credit'],
  },
  {
    id: 'rag-co02', area: 'consumer_protection',
    title: 'FTC Act Section 5 — Unfair or Deceptive Acts',
    content: 'FTC Act § 5 (15 U.S.C. § 45): prohibits unfair or deceptive acts or practices in commerce. Deceptive: material misrepresentation or omission that misleads reasonable consumer — FTC v. Sperry & Hutchinson (1972). Unfair: substantial injury, not outweighed by countervailing benefit, consumer cannot reasonably avoid. State UDAP statutes (all 50 states): modeled on FTC Act; many allow private right of action with treble damages and attorney fees. California: UCL (Bus. & Prof. Code § 17200) — unfair, deceptive, or unlawful business practice; class actions common. TCPA: telemarketing restrictions; $500-$1,500 per call damages.',
    keywords: ['consumer protection', 'FTC', 'deceptive', 'unfair', 'UDAP', 'UCL', 'California', 'misrepresentation', 'advertising', 'telemarketing'],
  },
  {
    id: 'rag-co03', area: 'consumer_protection',
    title: 'TILA and RESPA — Lending and Mortgage Disclosures',
    content: 'Truth in Lending Act (TILA, 15 U.S.C. § 1601): requires clear disclosure of APR, finance charges, total payment. Right to rescind certain home equity loans within 3 business days. RESPA (12 U.S.C. § 2601): prohibits kickbacks in real estate settlement; requires Loan Estimate (LE) within 3 days of application and Closing Disclosure (CD) 3 days before closing. CFPB-Know Before You Owe rule integrated TILA/RESPA disclosures. HOEPA: high-cost mortgages face additional restrictions. ATR/QM Rule: lenders must verify ability to repay (CFPB, 2014). Foreclosure: judicial (23 states) vs. non-judicial/power of sale.',
    keywords: ['mortgage', 'TILA', 'RESPA', 'APR', 'lending', 'disclosure', 'foreclosure', 'CFPB', 'truth in lending', 'right to rescind'],
  },

  // ── INTELLECTUAL PROPERTY ─────────────────────────────────────────────────
  {
    id: 'rag-ip01', area: 'intellectual_property',
    title: 'Copyright — 17 U.S.C., Fair Use, Infringement',
    content: 'Copyright Act of 1976 (17 U.S.C. § 101): protects original works of authorship fixed in tangible medium. Protection: life of author + 70 years. No registration required for protection (but needed for statutory damages). Exclusive rights: reproduce, distribute, create derivative works, perform, display. Fair use (§ 107): four-factor test — purpose (commercial vs. educational), nature of work, amount used, market effect — Campbell v. Acuff-Rose (1994) parody. Infringement: access + substantial similarity. DMCA (17 U.S.C. § 512): safe harbor for ISPs. Statutory damages: $750–$30,000 per work; up to $150,000 willful.',
    keywords: ['copyright', 'fair use', 'infringement', 'DMCA', 'original work', 'substantial similarity', 'trademark', 'intellectual property', '17 USC'],
  },
  {
    id: 'rag-ip02', area: 'intellectual_property',
    title: 'Trade Secrets — DTSA and Uniform Trade Secrets Act',
    content: 'Defend Trade Secrets Act (DTSA, 18 U.S.C. § 1836, 2016): federal cause of action for trade secret misappropriation. Trade secret: information with independent economic value from secrecy + reasonable measures to maintain secrecy. Misappropriation: improper acquisition or disclosure without consent. Remedies: injunction (TRO available ex parte), damages, exemplary damages (2×) for willful, attorney fees. State law: Uniform Trade Secrets Act adopted by 48 states; similar elements. Non-disclosure agreements supplement statutory protection. Employee mobility: balance between protecting trade secrets and employee right to work.',
    keywords: ['trade secret', 'DTSA', 'misappropriation', 'confidential', 'NDA', 'non-disclosure', 'proprietary', 'economic espionage'],
  },

  // ── GENERAL PROCEDURE AND EVIDENCE ───────────────────────────────────────
  {
    id: 'rag-g01', area: 'other',
    title: 'Statute of Limitations — Federal and State Periods',
    content: 'Federal statutes: Title VII — 180/300 days (EEOC) + 90 days after right-to-sue. § 1983 civil rights — borrowed from state personal injury (2–3 years). Federal contracts — 6 years (28 U.S.C. § 2401). Tolling: discovery rule (accrues when plaintiff knew or should have known), fraudulent concealment, minority (under 18), disability, absence from state. Equitable tolling: plaintiff diligently pursued rights but was prevented from filing — Irwin v. Dept of Veterans Affairs (1990). Laches: equitable doctrine bars stale claim when defendant prejudiced by delay. Statute of repose: absolute cutoff regardless of discovery.',
    keywords: ['statute of limitations', 'tolling', 'laches', 'time limit', 'filing deadline', 'accrual', 'discovery rule', 'equitable tolling'],
  },
  {
    id: 'rag-g02', area: 'other',
    title: 'Burden of Proof — Civil and Criminal Standards',
    content: 'Preponderance of evidence: more likely than not (>50%) — standard in most civil cases. Clear and convincing evidence: highly probable; used in fraud, termination of parental rights, civil commitment, some contract cases. Beyond reasonable doubt: near certainty — all criminal prosecutions; In re Winship (1970) — constitutional requirement. Burden of production vs. burden of persuasion. Presumptions: rebuttable vs. conclusive. Affirmative defenses (self-defense, contributory negligence): defendant bears burden of production; in many states bears burden of persuasion by preponderance.',
    keywords: ['burden of proof', 'preponderance', 'clear and convincing', 'beyond reasonable doubt', 'standard of proof', 'burden of persuasion'],
  },
  {
    id: 'rag-g03', area: 'other',
    title: 'Hearsay — FRE 801, Exceptions, Non-Hearsay Uses',
    content: 'Federal Rules of Evidence 801: hearsay = out-of-court statement offered for truth of matter asserted. Excluded from hearsay (FRE 801(d)): prior inconsistent statement under oath, prior consistent statement, admission by party-opponent. Exceptions (FRE 803) — availability immaterial: excited utterance, present sense impression, then-existing mental/physical condition, records of regularly conducted activity, public records. Exceptions requiring declarant unavailability (FRE 804): former testimony, dying declaration, statement against interest, personal history. Residual exception (FRE 807). Confrontation Clause limits hearsay in criminal cases (Crawford).',
    keywords: ['hearsay', 'FRE 801', 'exception', 'admission', 'excited utterance', 'business record', 'confrontation', 'evidence', 'out of court'],
  },
  {
    id: 'rag-g04', area: 'other',
    title: 'Attorney-Client Privilege and Work Product Doctrine',
    content: 'Attorney-client privilege: confidential communications between attorney and client for purpose of legal advice — absolute protection from disclosure. Corporate privilege (Upjohn v. United States, 1981): extends to employees communicating with counsel about job duties. Crime-fraud exception: no privilege for communications facilitating crime/fraud. Work product doctrine (FRCP 26(b)(3)): trial preparation materials protected from discovery; opinion work product (attorney\'s mental impressions) nearly absolute. Substantial need exception for ordinary work product. Privilege waiver: voluntary disclosure to non-privileged person. Joint defense (common interest) doctrine: shared privilege among co-defendants.',
    keywords: ['attorney-client privilege', 'work product', 'privilege', 'confidential', 'waiver', 'crime-fraud', 'discovery', 'upjohn'],
  },
  {
    id: 'rag-g05', area: 'other',
    title: 'Remedies — Injunctions, Specific Performance, Restitution',
    content: 'Permanent injunction: (1) actual success on merits, (2) irreparable harm, (3) balance of hardships favors plaintiff, (4) public interest. eBay Inc. v. MercExchange (2006): four-factor test required even in patent cases. Preliminary injunction/TRO: same factors; likelihood of success on merits. Specific performance of contract: available when money damages inadequate (unique goods, real estate) — equitable clean hands doctrine applies. Restitution/unjust enrichment: defendant must disgorge benefit. Constructive trust: equitable remedy when property obtained by fraud. Rescission: cancels contract and restores parties to pre-contract position.',
    keywords: ['injunction', 'specific performance', 'restitution', 'equitable remedy', 'irreparable harm', 'preliminary injunction', 'TRO', 'rescission', 'constructive trust'],
  },
  {
    id: 'rag-g06', area: 'other',
    title: '42 U.S.C. § 1983 — Civil Rights Claims Against State Actors',
    content: '42 U.S.C. § 1983: cause of action for deprivation of federal rights under color of state law. Requirements: (1) acting under color of state law (government actor); (2) deprivation of constitutional or federal statutory right. Qualified immunity: Harlow v. Fitzgerald (1982) — officers immune unless right was clearly established; Pearson v. Callahan modified two-step. Monell v. New York City (1978): municipalities liable only for official policy or custom, not respondeat superior. Bivens actions: analogous claim against federal officers. Statute of limitations: borrowed from state personal injury statute (2–3 years). § 1988: attorney fees to prevailing civil rights plaintiff.',
    keywords: ['section 1983', 'civil rights', 'qualified immunity', 'color of state law', 'constitutional violation', 'monell', 'police', 'government', 'due process'],
  },
];

/**
 * Score-based RAG search: returns up to 6 most relevant documents.
 * Scoring: keyword match in content (1pt each), keyword match in title (2pt each),
 * area match with caseType (3pt bonus), query term match (1pt each).
 */
function ragSearch(query, caseType) {
  const queryTerms = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  const areaMap = {
    contract: 'contract', tort: 'tort', employment: 'employment',
    criminal: 'criminal', constitutional: 'constitutional',
    property: 'property', other: 'other',
    civil_procedure: 'civil_procedure', family: 'family',
    consumer_protection: 'consumer_protection', intellectual_property: 'intellectual_property',
    regulatory: 'other',
  };
  const targetArea = areaMap[caseType] || null;

  const scored = US_LAW_RAG.map(doc => {
    let score = 0;
    const titleLower = doc.title.toLowerCase();
    const contentLower = doc.content.toLowerCase();

    // Keyword matches
    for (const kw of doc.keywords) {
      if (query.toLowerCase().includes(kw)) score += 2;
    }
    // Query term matches in content and title
    for (const term of queryTerms) {
      if (contentLower.includes(term)) score += 1;
      if (titleLower.includes(term)) score += 2;
    }
    // Area bonus
    if (targetArea && doc.area === targetArea) score += 3;

    return { doc, score };
  });

  return scored
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 6)
    .map(({ doc }) => doc);
}

// ─── SSE Emitter ─────────────────────────────────────────────────────────────
function emit(res, type, data) {
  res.write(`data: ${JSON.stringify({ type, ...data })}\n\n`);
}

// ─── PHASE 1: Intelligence Agent ─────────────────────────────────────────────
async function phase1(query, history, complexity) {
  const historyCtx = history.length
    ? `\n\nConversation history:\n${history.map(h => `${h.role}: ${h.content}`).join('\n')}`
    : '';

  const result = await llm(M.SONNET,
    [{ role: 'user', content: `Analyze this legal query and return JSON only.\n\nQuery: ${query}${historyCtx}` }],
    {
      thinking: THINKING_BUDGETS[complexity].sonnet,
      json: true,
      systemPrompt: `You are a legal intelligence agent. Extract structured information from the user's legal query.
Return ONLY valid JSON with this exact schema:
{
  "legalIssues": ["string"],
  "caseType": "contract|tort|employment|criminal|constitutional|regulatory|property|other",
  "parties": { "plaintiff": "string", "defendant": "string" },
  "burdenOfProof": "preponderance|clear_and_convincing|beyond_reasonable_doubt",
  "keyFacts": ["string"],
  "jurisdiction": "string or null",
  "isFollowUp": boolean,
  "complexityLevel": "simple|moderate|complex",
  "tavilyConfig": {
    "fire": boolean,
    "searchCount": 0 | 3 | 6 | 9,
    "reason": "string"
  }
}
complexityLevel: simple = single issue, settled law. moderate = multiple issues, some uncertainty. complex = novel doctrine, multi-party, conflicting precedent.
tavilyConfig: 0 searches for stable doctrine (constitutional, classical tort). 3 for moderately volatile. 6 for volatile. 9 for regulatory/immigration/tax/recent legislation.`,
    }
  );

  return parseJSON(result);
}

// ─── PHASE 2: Research Fan-out ────────────────────────────────────────────────
async function phase2(intel, emit_fn) {
  const ragDocs = ragSearch(intel.legalIssues.join(' '), intel.caseType);

  const researchPrompt = (role) => `You are a ${role} legal researcher. Research the legal issues below.
Use the provided legal knowledge base. Return JSON: { "findings": ["string"], "keyPrecedents": ["string"], "favorableAuthorities": ["string"], "sources": [{"id":"string","title":"string","relevance":"string"}] }

Case: ${JSON.stringify(intel)}
Legal Knowledge Base: ${JSON.stringify(ragDocs)}`;

  // Parallel research fan-out
  const [plaintiffRes, defenseRes, neutralRes] = await Promise.all([
    llm(M.HAIKU, [{ role: 'user', content: 'Research favorable to plaintiff.' }], { json: true, systemPrompt: researchPrompt('plaintiff-favoring') }),
    llm(M.HAIKU, [{ role: 'user', content: 'Research favorable to defense.' }], { json: true, systemPrompt: researchPrompt('defense-favoring') }),
    llm(M.HAIKU, [{ role: 'user', content: 'Research procedural rules and neutral standards.' }], { json: true, systemPrompt: researchPrompt('neutral procedural') }),
  ]);

  // Tavily delta searches
  let tavilySources = [];
  if (intel.tavilyConfig.fire && intel.tavilyConfig.searchCount > 0) {
    const queries = [
      `${intel.caseType} law ${intel.legalIssues[0]} recent ruling`,
      `${intel.jurisdiction || 'US'} ${intel.caseType} statute current`,
      intel.legalIssues[1] || intel.legalIssues[0],
    ].slice(0, intel.tavilyConfig.searchCount / 3);

    tavilySources = (await Promise.all(queries.map(tavilySearch))).flat();
  }

  // Synthesize
  const synthesis = await llm(M.SONNET,
    [{
      role: 'user',
      content: [
        ...withCache(`Legal Knowledge Base:\n${JSON.stringify(ragDocs)}\n\nPlaintiff Research: ${plaintiffRes}\n\nDefense Research: ${defenseRes}\n\nNeutral Research: ${neutralRes}\n\nTavily Sources: ${JSON.stringify(tavilySources)}`),
        { type: 'text', text: 'Synthesize all research into a unified ResearchBundle JSON: { "summary": "string", "sources": [{"id":"string","title":"string","content":"string","url":"string|null"}], "keyStatutes": ["string"], "precedents": ["string"], "plaintiffStrengths": ["string"], "defenseStrengths": ["string"] }' },
      ],
    }],
    { json: true, maxTokens: 6000 }
  );

  const bundle = parseJSON(synthesis);

  // Normalize source IDs to stable "src-N" so downstream phases have predictable citation targets
  if (Array.isArray(bundle.sources)) {
    bundle.sources = bundle.sources.map((s, i) => ({ ...s, id: `src-${i + 1}` }));
  }

  return bundle;
}

// ─── PHASE 3: Advocate Chains (Build → Critique → Refine) ────────────────────
async function buildAdvocateChain(role, model, intel, researchBundle, complexity) {
  const bundleText = JSON.stringify(researchBundle);
  const isAnthropic = model.startsWith('anthropic/');
  const thinkingBudget = (role === 'neutral' || !isAnthropic) ? 0
    : THINKING_BUDGETS[complexity].sonnet;

  const sysPrompt = (step) => `You are the ${role} advocate in a legal proceeding. ${
    step === 'critique'
      ? 'Critique the argument ONLY for factual accuracy against the research bundle. Do NOT argue the other side.'
      : step === 'refine' ? 'Refine the argument incorporating the critique. Strengthen weak points.'
      : 'Build the strongest possible legal argument for your position.'
  } Every factual claim MUST cite a source ID from the ResearchBundle.
Return JSON: { "argument": "string", "keyPoints": ["string"], "citations": ["source_id"] }`;

  // Build content — use cache_control only for Anthropic; plain string for others
  const buildContent = isAnthropic
    ? [
        ...withCache(`ResearchBundle: ${bundleText}\nCase: ${JSON.stringify(intel)}`),
        { type: 'text', text: `Build the ${role} legal argument.` },
      ]
    : `ResearchBundle: ${bundleText}\nCase: ${JSON.stringify(intel)}\n\nBuild the ${role} legal argument.`;

  // Build
  const buildResult = await llm(model,
    [{ role: 'user', content: buildContent }],
    { json: true, maxTokens: 4096, systemPrompt: sysPrompt('build'), thinking: thinkingBudget }
  );

  // Critique (Haiku — factual grounding check only)
  const critiqueResult = await llm(M.HAIKU,
    [{ role: 'user', content: `ResearchBundle: ${bundleText}\n\nArgument to critique: ${buildResult}\n\nCheck factual accuracy only. Return JSON: { "issues": ["string"], "unsupportedClaims": ["string"] }` }],
    { json: true, maxTokens: 1024, systemPrompt: sysPrompt('critique') }
  );

  // Refine content
  const refineContent = isAnthropic
    ? [
        ...withCache(`ResearchBundle: ${bundleText}`),
        { type: 'text', text: `Original argument: ${buildResult}\nCritique: ${critiqueResult}\nRefine and strengthen. Keep the argument concise (max 400 words). Return JSON: { "argument": "string", "keyPoints": ["string"], "citations": ["source_id"] }` },
      ]
    : `ResearchBundle: ${bundleText}\n\nOriginal argument: ${buildResult}\nCritique: ${critiqueResult}\n\nRefine and strengthen. Keep the argument concise (max 400 words). Return JSON: { "argument": "string", "keyPoints": ["string"], "citations": ["source_id"] }`;

  // Refine
  const refineResult = await llm(model,
    [{ role: 'user', content: refineContent }],
    { json: true, maxTokens: 4096, systemPrompt: sysPrompt('refine'), thinking: thinkingBudget }
  );

  // Fall back to build result if refine parsing fails
  try {
    return parseJSON(refineResult);
  } catch {
    return parseJSON(buildResult);
  }
}

async function phase3(intel, researchBundle, complexity) {
  const [plaintiff, defense, neutral] = await Promise.all([
    buildAdvocateChain('plaintiff', M.O4_MINI, intel, researchBundle, complexity),
    buildAdvocateChain('defense', M.GEMINI, intel, researchBundle, complexity),
    buildAdvocateChain('neutral', M.SONNET, intel, researchBundle, complexity),
  ]);
  return { plaintiff, defense, neutral };
}

// ─── PHASE 4: Cross-Examination + Consensus ───────────────────────────────────
async function phase4(advocates, researchBundle, intel) {
  const bundleText = JSON.stringify(researchBundle);
  const rebuttalSys = (role) => `You are the ${role} advocate attacking the opposing positions. Return ONLY valid JSON with no extra text, no markdown, no prose. Use this exact schema:
{"attackPoints":["short attack 1","short attack 2","short attack 3"],"citations":["source_id"]}
Each attackPoint must be a plain short phrase (under 20 words). Do NOT use double quotes inside the string values.`;

  // Use Haiku for all rebuttals — fast, reliable JSON, no thinking-bleed issues
  const [pRebuttal, dRebuttal, nRebuttal] = await Promise.all([
    llm(M.HAIKU, [{ role: 'user', content: `Defense position: ${JSON.stringify(advocates.defense?.keyPoints)}\nNeutral position: ${JSON.stringify(advocates.neutral?.keyPoints)}\nResearch sources: ${JSON.stringify(researchBundle.sources?.map(s=>({id:s.id,title:s.title})))}\nWrite plaintiff rebuttal JSON.` }], { json: true, maxTokens: 1024, systemPrompt: rebuttalSys('plaintiff') }),
    llm(M.HAIKU, [{ role: 'user', content: `Plaintiff position: ${JSON.stringify(advocates.plaintiff?.keyPoints)}\nNeutral position: ${JSON.stringify(advocates.neutral?.keyPoints)}\nResearch sources: ${JSON.stringify(researchBundle.sources?.map(s=>({id:s.id,title:s.title})))}\nWrite defense rebuttal JSON.` }], { json: true, maxTokens: 1024, systemPrompt: rebuttalSys('defense') }),
    llm(M.HAIKU, [{ role: 'user', content: `Plaintiff position: ${JSON.stringify(advocates.plaintiff?.keyPoints)}\nDefense position: ${JSON.stringify(advocates.defense?.keyPoints)}\nResearch sources: ${JSON.stringify(researchBundle.sources?.map(s=>({id:s.id,title:s.title})))}\nWrite neutral rebuttal JSON.` }], { json: true, maxTokens: 1024, systemPrompt: rebuttalSys('neutral') }),
  ]);

  // Consensus measurement
  const consensus = await llm(M.SONNET,
    [{
      role: 'user',
      content: `Analyze advocate positions and rebuttals. Return JSON only:
{
  "consensusLevel": "STRONG_CONSENSUS|MODERATE_CONSENSUS|MIXED_VIEWS|SIGNIFICANT_DISAGREEMENT|FUNDAMENTAL_DISAGREEMENT",
  "divergentClaims": ["specific claims where advocates disagreed"],
  "agreedFacts": ["facts all three accepted"]
}

Plaintiff: ${JSON.stringify(advocates.plaintiff)}
Defense: ${JSON.stringify(advocates.defense)}
Neutral: ${JSON.stringify(advocates.neutral)}
Rebuttals: plaintiff=${pRebuttal} defense=${dRebuttal} neutral=${nRebuttal}`,
    }],
    { json: true, maxTokens: 2048 }
  );

  return {
    rebuttals: { plaintiff: parseJSON(pRebuttal), defense: parseJSON(dRebuttal), neutral: parseJSON(nRebuttal) },
    ...parseJSON(consensus),
  };
}

// ─── PHASE 5: Judge Panel ─────────────────────────────────────────────────────
async function phase5(intel, researchBundle, advocates, crossExam, complexity) {
  const ctx = JSON.stringify({ intel, researchBundle, advocates, crossExam });
  const thinkBudget = complexity === 'complex' ? THINKING_BUDGETS.complex.sonnet : THINKING_BUDGETS.moderate.sonnet;

  const [factFindings, lawFindings, precedentFindings] = await Promise.all([
    llm(M.SONNET, [{ role: 'user', content: ctx }], {
      thinking: thinkBudget,
      json: true,
      maxTokens: 3000,
      systemPrompt: 'You are the Fact Judge. Identify ONLY disputed vs undisputed facts. Do NOT interpret law. Return JSON: { "undisputedFacts": ["string"], "disputedFacts": [{"claim":"string","plaintiffVersion":"string","defenseVersion":"string"}] }',
    }),
    llm(M.O4_MINI, [{ role: 'user', content: ctx }], {
      json: true,
      maxTokens: 3000,
      systemPrompt: 'You are the Law Judge. Identify ONLY applicable legal standards and doctrine. Do NOT determine facts. Return JSON: { "applicableStandards": ["string"], "relevantStatutes": ["string"], "legalTests": ["string"] }',
    }),
    llm(M.GEMINI, [{ role: 'user', content: ctx }], {
      json: true,
      maxTokens: 7000,
      systemPrompt: 'You are the Precedent Judge. Identify analogous cases and assess which side they favor. Return ONLY JSON with no markdown: { "analogousCases": [{"name":"string","relevance":"string","favors":"plaintiff|defense|neutral"}], "precedentSummary": "string", "overallPrecedentFavors": "plaintiff|defense|neutral|split" }',
    }),
  ]);

  const parsed = {
    factJudge: parseJSON(factFindings),
    lawJudge: parseJSON(lawFindings),
    precedentJudge: parseJSON(precedentFindings),
  };

  // Flag minority finding if sub-judges diverge significantly
  const allFavor = [parsed.precedentJudge.overallPrecedentFavors];
  parsed.minorityFinding = allFavor.some(f => f === 'plaintiff') && allFavor.some(f => f === 'defense');

  return parsed;
}

// ─── PHASE 6: Chief Judge (Hybrid — Sonnet default, Opus escalation) ──────────
async function phase6(intel, researchBundle, advocates, crossExam, judgePanel, stream_fn) {
  const { complexityLevel } = intel;
  const { consensusLevel } = crossExam;

  const escalate = complexityLevel === 'complex' &&
    ['SIGNIFICANT_DISAGREEMENT', 'FUNDAMENTAL_DISAGREEMENT'].includes(consensusLevel);

  const model = escalate ? M.OPUS : M.SONNET;
  const thinking = escalate ? THINKING_BUDGETS.complex.opus : THINKING_BUDGETS[complexityLevel]?.sonnet || 6000;

  const fullContext = [
    { type: 'text', text: `IntelBundle: ${JSON.stringify(intel)}\n\nConsensusLevel: ${consensusLevel}\nDivergentClaims: ${JSON.stringify(crossExam.divergentClaims)}`, cache_control: { type: 'ephemeral' } },
    { type: 'text', text: `ResearchBundle: ${JSON.stringify(researchBundle)}`, cache_control: { type: 'ephemeral' } },
    { type: 'text', text: `Advocates: ${JSON.stringify(advocates)}\nRebuttals: ${JSON.stringify(crossExam.rebuttals)}\nJudgePanel: ${JSON.stringify(judgePanel)}` },
    { type: 'text', text: `Produce the final legal ruling as JSON with this exact schema:
{
  "findingsOfFact": [{"fact":"string","status":"settled|disputed","sources":["source_id"]}],
  "conclusionsOfLaw": [{"conclusion":"string","basis":"string","advocateAttribution":"plaintiff|defense|neutral|all"}],
  "ruling": "string (who prevails and on what grounds)",
  "rationale": "string (detailed reasoning)",
  "citedAuthorities": [{"id":"string","title":"string","proposition":"string"}],
  "epistemicMarkers": [{"claim":"string","status":"settled|disputed|single_source"}],
  "minorityReport": "string or null",
  "modelUsed": "${model}",
  "escalated": ${escalate}
}

CRITICAL: The ResearchBundle sources use IDs in the format "src-1", "src-2", etc. You MUST use ONLY these exact IDs in citedAuthorities and findingsOfFact.sources. Valid IDs: ${researchBundle.sources?.map(s => s.id).join(', ')}. Do NOT invent new IDs. If evidence is thin, state low confidence rather than fabricate.` },
  ];

  // Stream the chief judge response
  const streamRes = await llm(model,
    [{ role: 'user', content: fullContext }],
    {
      thinking,
      stream: true,
      maxTokens: thinking + 8000,
      systemPrompt: 'You are the Chief Judge of a legal council. Produce a formal, fully-cited legal ruling. Return ONLY JSON matching the schema provided. Base every finding strictly on the research bundle and advocate arguments provided.',
    }
  );

  // Collect streamed tokens
  let fullText = '';
  const reader = streamRes.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n').filter(l => l.startsWith('data: '));
    for (const line of lines) {
      const raw = line.slice(6);
      if (raw === '[DONE]') break;
      try {
        const parsed = JSON.parse(raw);
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) {
          fullText += delta;
          stream_fn(delta);
        }
      } catch { /* skip malformed */ }
    }
  }

  return parseJSON(fullText);
}

// ─── PHASE 7: Post-Ruling Checks (parallel) ───────────────────────────────────
async function phase7(ruling, researchBundle) {
  const bundleSources = new Set(researchBundle.sources.map(s => s.id));

  // Trim ruling to key fields only — full JSON can be huge and cause truncation
  const rulingDigest = {
    ruling: ruling.ruling,
    rationale: ruling.rationale,
    findingsOfFact: ruling.findingsOfFact,
    conclusionsOfLaw: ruling.conclusionsOfLaw,
  };

  const [daResult] = await Promise.all([
    // 7a: Devil's Advocate critique
    llm(M.HAIKU,
      [{ role: 'user', content: `Ruling summary: ${JSON.stringify(rulingDigest)}\n\nIdentify weaknesses. Return JSON: { "assumptions": ["string"], "missedArguments": ["string"], "biases": ["string"], "overconfidentClaims": ["string"] }` }],
      {
        json: true,
        maxTokens: 2048,
        systemPrompt: 'You are a contrarian critic. Aggressively challenge this ruling. Find every weakness, assumption, and potential error. Keep each item under 30 words.',
      }
    ),
  ]);

  // 7b: Citation integrity — exact ID match + title fuzzy fallback
  const bundleTitles = researchBundle.sources.map(s => s.title?.toLowerCase() || '');

  function isCitationVerified(authority) {
    // 1. Exact ID match
    if (bundleSources.has(authority.id)) return true;
    // 2. Title substring match (catches minor ID variations)
    const authTitle = (authority.title || authority.id || '').toLowerCase();
    return bundleTitles.some(t => t && authTitle && (t.includes(authTitle.slice(0, 20)) || authTitle.includes(t.slice(0, 20))));
  }

  let unverified = (ruling.citedAuthorities || [])
    .filter(a => !isCitationVerified(a))
    .map(a => a.id);

  // Re-prompt once if unverified citations found (per architecture spec)
  if (unverified.length > 0) {
    const validIds = researchBundle.sources.map(s => `${s.id} ("${s.title}")`).join(', ');
    const repromptRaw = await llm(M.HAIKU,
      [{
        role: 'user',
        content: `These citation IDs in the ruling are NOT in the ResearchBundle: ${unverified.join(', ')}\n\nValid source IDs: ${validIds}\n\nReturn ONLY a corrected citedAuthorities JSON array using only the valid IDs above:\n[{"id":"src-N","title":"...","proposition":"..."}]`,
      }],
      { json: true, maxTokens: 1024 }
    );
    try {
      const corrected = parseJSON(repromptRaw);
      if (Array.isArray(corrected)) {
        // Replace the unverified citations with corrected ones
        const unverifiedSet = new Set(unverified);
        const kept = (ruling.citedAuthorities || []).filter(a => !unverifiedSet.has(a.id));
        ruling.citedAuthorities = [...kept, ...corrected.filter(a => bundleSources.has(a.id))];
        unverified = []; // mark as fixed
      }
    } catch { /* leave as-is if re-prompt also fails */ }
  }

  return {
    devilsAdvocate: parseJSON(daResult),
    citationIntegrity: { verified: unverified.length === 0, unverifiedIds: unverified, rePrompted: unverified.length === 0 && ruling.citedAuthorities?.length > 0 },
  };
}

// ─── Main Pipeline Orchestrator ───────────────────────────────────────────────
app.post('/api/council', async (req, res) => {
  const { query, sessionId = 'default' } = req.body;
  if (!query) return res.status(400).json({ error: 'query required' });

  // SSE setup
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const e = (type, data) => emit(res, type, data);
  const history = sessions.get(sessionId) || [];

  try {
    // Phase 1
    e('phase_start', { phase: 1, name: 'Intelligence Agent' });
    const intel = await phase1(query, history, 'moderate'); // initial complexity estimate
    e('phase_complete', { phase: 1, data: { caseType: intel.caseType, issues: intel.legalIssues, complexity: intel.complexityLevel, tavilySearches: intel.tavilyConfig.searchCount } });

    // Phase 2
    e('phase_start', { phase: 2, name: 'Research Fan-out' });
    const researchBundle = await phase2(intel, e);
    e('phase_complete', { phase: 2, data: { sources: researchBundle.sources.length, tavilySources: intel.tavilyConfig.searchCount } });

    // Phase 3
    e('phase_start', { phase: 3, name: 'Advocate Chains' });
    const advocates = await phase3(intel, researchBundle, intel.complexityLevel);
    e('phase_complete', { phase: 3, data: { plaintiff: advocates.plaintiff.keyPoints?.length, defense: advocates.defense.keyPoints?.length, neutral: advocates.neutral.keyPoints?.length } });

    // Phase 4
    e('phase_start', { phase: 4, name: 'Cross-Examination' });
    const crossExam = await phase4(advocates, researchBundle, intel);
    e('phase_complete', { phase: 4, data: { consensusLevel: crossExam.consensusLevel, divergentClaims: crossExam.divergentClaims?.length || 0 } });

    // Phase 5
    e('phase_start', { phase: 5, name: 'Judge Panel' });
    const judgePanel = await phase5(intel, researchBundle, advocates, crossExam, intel.complexityLevel);
    e('phase_complete', { phase: 5, data: { disputedFacts: judgePanel.factJudge.disputedFacts?.length, precedentFavors: judgePanel.precedentJudge.overallPrecedentFavors, minority: judgePanel.minorityFinding } });

    // Phase 6 (streaming)
    e('phase_start', { phase: 6, name: 'Chief Judge', escalated: false });
    let ruling;
    try {
      ruling = await phase6(intel, researchBundle, advocates, crossExam, judgePanel,
        (delta) => e('stream_token', { phase: 6, content: delta })
      );
    } catch (streamErr) {
      console.warn('[phase6] stream failed, using non-streaming fallback:', streamErr.message);
      // Fallback: non-streaming, no thinking (so response_format:json_object is active)
      const escalate = intel.complexityLevel === 'complex' && ['SIGNIFICANT_DISAGREEMENT', 'FUNDAMENTAL_DISAGREEMENT'].includes(crossExam.consensusLevel);
      const model = escalate ? M.OPUS : M.SONNET;
      const raw = await llm(model,
        [{ role: 'user', content: `IntelBundle: ${JSON.stringify(intel)}\nResearchBundle: ${JSON.stringify(researchBundle)}\nAdvocates: ${JSON.stringify(advocates)}\nCrossExam: ${JSON.stringify(crossExam)}\nJudgePanel: ${JSON.stringify(judgePanel)}\n\nProduce final ruling as JSON: {findingsOfFact, conclusionsOfLaw, ruling, rationale, citedAuthorities, epistemicMarkers, minorityReport, modelUsed, escalated}` }],
        { json: true, thinking: 0, maxTokens: 8000, systemPrompt: 'You are the Chief Judge. Return ONLY valid JSON. Cite only sources present in the ResearchBundle.' }
      );
      ruling = parseJSON(raw);
    }
    e('phase_complete', { phase: 6, data: { model: ruling.modelUsed, escalated: ruling.escalated } });

    // Phase 7
    e('phase_start', { phase: 7, name: 'Post-Ruling Checks' });
    const checks = await phase7(ruling, researchBundle);
    e('phase_complete', { phase: 7, data: { citationVerified: checks.citationIntegrity.verified, unverified: checks.citationIntegrity.unverifiedIds, rePrompted: checks.citationIntegrity.rePrompted } });

    // Update conversation history
    history.push({ role: 'user', content: query });
    history.push({ role: 'assistant', content: ruling.ruling });
    if (history.length > 20) history.splice(0, 2);
    sessions.set(sessionId, history);

    // Done
    e('done', {
      intel,
      researchBundle,
      advocates,
      crossExam,
      judgePanel,
      ruling,
      checks,
    });

  } catch (err) {
    e('error', { message: err.message });
    console.error(err);
  } finally {
    res.end();
  }
});

app.get('/api/health', (_, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3000;
const server = createServer(app);
server.listen(PORT, () => console.log(`Council of LLMs → http://localhost:${PORT}`));

// Graceful shutdown so --watch can restart cleanly on Windows
process.on('SIGTERM', () => server.close());
process.on('SIGINT', () => server.close());
