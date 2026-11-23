import { describe, expect, it } from "vitest";
import { readLocalProfile } from "./csvProfile";
describe("local CSV profiling", () => {
  it("honors quoted commas, calculates missingness, and keeps identifiers out of class labels", () => {
    const result = readLocalProfile(
      'customer_id,target,description\nA,yes,"a,b"\nB,no,\nC,yes,c\n',
    );
    expect(result.profile.rows).toBe(3);
    expect(result.profile.missing.description).toBeCloseTo(33.33, 2);
    expect(result.profile.warnings[0]).toContain("customer_id");
    expect(result.balanceFor("target")).toEqual({ yes: 2, no: 1 });
    expect(result.balanceFor("customer_id")).toEqual({});
  });
  it("rejects inconsistent rows and duplicate headers", () => {
    expect(() => readLocalProfile("a,b\n1,2\n3")).toThrow();
    expect(() => readLocalProfile("a,a\n1,2")).toThrow();
  });
});
