{
  description = "Real-time multi-physics simulation with an emphasis on medical simulation.";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # ref. https://github.com/NixOS/nixpkgs/pull/348549
    nixpkgs.url = "github:nim65s/nixpkgs/qt6-libqglviewer";
  };

  outputs =
    inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', ... }:
        {
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.sofa;
            sofa = pkgs.callPackage ./package.nix { };
          };
        };
    };
}
