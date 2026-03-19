"""Generate synthetic Needle-in-a-Haystack (NIAH) training data.

A factual "needle" is hidden among filler paragraphs; the model must retrieve it.

Usage:
    python scripts/generate_niah_data.py --num-train 500 --num-val 50
"""

import argparse
import random
from lib.io import save_jsonl

FILLER_PARAGRAPHS = [
    "The history of agriculture dates back thousands of years. Early humans transitioned from hunter-gatherer societies to settled farming communities around 10,000 BCE. This shift, known as the Neolithic Revolution, fundamentally changed human civilization. Crops like wheat, barley, and rice were among the first to be domesticated.",
    "Ocean currents play a crucial role in regulating Earth's climate. The Gulf Stream, for example, carries warm water from the Gulf of Mexico across the Atlantic, moderating temperatures in Western Europe. Without these currents, many coastal regions would experience much more extreme weather patterns.",
    "The development of the printing press by Johannes Gutenberg around 1440 revolutionized the spread of information. Before this invention, books were painstakingly copied by hand, making them expensive and rare. The printing press enabled mass production of written works and helped fuel the Renaissance.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This biological process is fundamental to life on Earth, as it produces the oxygen we breathe and forms the base of most food chains in terrestrial ecosystems.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries. Contrary to popular belief, it is not visible from space with the naked eye. The wall was constructed primarily to protect against invasions from northern nomadic groups.",
    "Mathematics has been called the language of the universe. From the geometric patterns in snowflakes to the fibonacci sequences in sunflower spirals, mathematical principles appear throughout nature. Ancient civilizations independently developed mathematical systems to track seasons and manage trade.",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. This vast neural network enables everything from basic motor functions to complex abstract thought. Despite decades of research, many aspects of brain function remain poorly understood.",
    "Volcanoes form when magma from deep within the Earth's mantle rises to the surface. The Ring of Fire, a horseshoe-shaped zone around the Pacific Ocean, contains about 75% of the world's active volcanoes. Volcanic eruptions can have both destructive and constructive effects on the landscape.",
    "The Renaissance, which began in Italy in the 14th century, was a period of renewed interest in classical art, literature, and learning. Artists like Leonardo da Vinci and Michelangelo produced works that continue to influence art today. The movement eventually spread throughout Europe.",
    "Water covers about 71% of the Earth's surface, but only about 2.5% of it is freshwater. Of that freshwater, roughly 69% is locked in glaciers and ice caps. Access to clean freshwater remains one of the most pressing global challenges of our time.",
    "The periodic table organizes chemical elements by their atomic number and chemical properties. Dmitri Mendeleev published the first widely recognized periodic table in 1869, predicting the existence of elements that had not yet been discovered. His framework remains foundational to chemistry.",
    "Ancient Rome was one of the largest empires in history, spanning three continents at its peak. Roman engineering achievements, including aqueducts, roads, and concrete construction, influenced infrastructure development for centuries. The fall of the Western Roman Empire in 476 CE marked a major turning point in European history.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen and houses an estimated 10% of all species on Earth. Deforestation threatens this vital ecosystem, with significant portions being cleared for agriculture and logging each year.",
    "Sound travels through air at approximately 343 meters per second at room temperature. It moves faster through liquids and solids because the molecules are closer together. Thunder is the sound produced by the rapid expansion of air heated by a lightning strike.",
    "The invention of the telescope in the early 17th century transformed astronomy. Galileo Galilei used an improved version to observe the moons of Jupiter, providing evidence for the heliocentric model of the solar system. Modern telescopes can detect light from billions of light-years away.",
    "Migration patterns in birds are among the most remarkable phenomena in the animal kingdom. Arctic terns travel from pole to pole each year, covering roughly 44,000 miles. These journeys are guided by a combination of the Earth's magnetic field, star positions, and memorized landmarks.",
]

NEEDLE_TEMPLATES = [
    ("The secret password for Project Aurora is 'crystal-phoenix-42'.",
     "What is the secret password for Project Aurora?",
     "The secret password for Project Aurora is 'crystal-phoenix-42'."),
    ("The treasure is buried under the old oak tree at coordinates 47.3N, 122.5W.",
     "Where is the treasure buried?",
     "The treasure is buried under the old oak tree at coordinates 47.3N, 122.5W."),
    ("Dr. Elena Vasquez discovered the high-temperature superconductor compound XR-7 in her lab on March 15th.",
     "Who discovered compound XR-7 and when?",
     "Dr. Elena Vasquez discovered compound XR-7 on March 15th."),
    ("The combination to the vault is 38-17-92-54.",
     "What is the combination to the vault?",
     "The combination to the vault is 38-17-92-54."),
    ("The next meeting of the International Council of Explorers will be held in Reykjavik on November 3rd.",
     "When and where is the next meeting of the International Council of Explorers?",
     "The next meeting will be held in Reykjavik on November 3rd."),
    ("Agent Nightingale's real name is Marcus Chen, and he operates out of Buenos Aires.",
     "What is Agent Nightingale's real name and where does he operate?",
     "Agent Nightingale's real name is Marcus Chen, and he operates out of Buenos Aires."),
    ("The antidote formula requires exactly 3.7 milliliters of compound Z mixed with 12 grams of sodium bicarbonate.",
     "What is the antidote formula?",
     "The antidote formula requires exactly 3.7 milliliters of compound Z mixed with 12 grams of sodium bicarbonate."),
    ("The launch code for satellite Omega-9 is 'THETA-LAMBDA-2847'.",
     "What is the launch code for satellite Omega-9?",
     "The launch code for satellite Omega-9 is 'THETA-LAMBDA-2847'."),
    ("The rare blue orchid can only be found in the caves of Mount Kaldera at an altitude of 2,340 meters.",
     "Where can the rare blue orchid be found?",
     "The rare blue orchid can only be found in the caves of Mount Kaldera at an altitude of 2,340 meters."),
    ("Professor Tanaka's final theorem states that prime numbers above 10^18 follow a spiral distribution pattern.",
     "What does Professor Tanaka's final theorem state?",
     "Professor Tanaka's final theorem states that prime numbers above 10^18 follow a spiral distribution pattern."),
]

INSTRUCTION = "Read the following passage carefully and answer the question based on the information provided."


def generate_example(num_paragraphs: int, rng: random.Random) -> dict:
    needle_fact, question, answer = rng.choice(NEEDLE_TEMPLATES)
    fillers = rng.sample(FILLER_PARAGRAPHS, min(num_paragraphs, len(FILLER_PARAGRAPHS)))
    pos = rng.randint(0, len(fillers))
    fillers.insert(pos, needle_fact)
    context = "\n\n".join(fillers)
    return {
        "instruction": INSTRUCTION,
        "input": f"Passage:\n{context}\n\nQuestion: {question}",
        "output": answer,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate NIAH training data")
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--min-paragraphs", type=int, default=5)
    parser.add_argument("--max-paragraphs", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for split, count in [("train", args.num_train), ("val", args.num_val)]:
        examples = [
            generate_example(rng.randint(args.min_paragraphs, args.max_paragraphs), rng)
            for _ in range(count)
        ]
        path = f"{args.output_dir}/niah_{split}.jsonl"
        save_jsonl(path, examples)
        print(f"Wrote {count} examples to {path}")


if __name__ == "__main__":
    main()
